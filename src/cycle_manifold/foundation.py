"""Codificador congelado (TimesFM u otro) con decodificador GRU entrenable.

El codificador convierte cada canal del ciclo en un vector fijo; una proyeccion
lineal entrenable produce z, que alimenta el decodificador refinado.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelSpec, TrainConfig, SEEDS
from .corruption import corrupt_batch
from .evaluation import evaluate
from .losses import reconstruction_loss
from .models import GRUDecoder
from .training import DEVICE, fit_model, set_seed, to_loader

TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"
N_PATCHES, PATCH_LEN = 4, 32  # 128 puntos = 4 parches de 32


class IdentityEncoder:
    """Control: la serie estandarizada aplanada, sin modelo preentrenado."""

    name = "identity"

    @torch.no_grad()
    def __call__(self, z):  # z: (B, C, T)
        return z.reshape(len(z), -1).float().cpu()


class TimesFMEncoder:
    """Estados ocultos de TimesFM 2.5 (200M) agrupados sobre los parches de cada canal.

    Cada canal se remuestrea a 128 puntos, se normaliza por instancia (RevIN) y
    atraviesa los bloques; del bloque elegido se promedian (o se toma el ultimo)
    los parches. Como RevIN elimina nivel y escala, se anaden media y desviacion.
    """

    def __init__(self, module, layers=(-1,), pooling="mean", add_stats=True, name="timesfm", chunk=256):
        self.m, self.add_stats, self.name, self.chunk = module, add_stats, name, chunk
        self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device = next(self.m.parameters()).device
        self._buffer = {}
        self.configure(layers, pooling)
        self._handles = [block.register_forward_hook(self._hook(i)) for i, block in enumerate(self.m.stacked_xf)]

    def configure(self, layers, pooling):
        """Cambia el bloque y la agrupacion sin recargar los pesos."""
        n_blocks = len(self.m.stacked_xf)
        self.idx, self.pooling = [layer % n_blocks for layer in layers], pooling

    def close(self):
        for handle in self._handles:
            handle.remove()

    def _hook(self, i):
        def store(module, inputs, output):
            if i in self.idx:
                self._buffer[i] = output[0]
        return store

    @torch.no_grad()
    def _series(self, s):  # s: (N, T) -> (N, d)
        from timesfm.torch import util  # import diferido: solo se necesita con TimesFM

        n = len(s)
        s = F.interpolate(s[:, None, :], size=N_PATCHES * PATCH_LEN, mode="linear", align_corners=True)[:, 0]
        patches = s.reshape(n, N_PATCHES, PATCH_LEN)
        mask = torch.zeros_like(patches, dtype=torch.bool)
        count, mu, sigma = (torch.zeros(n, device=s.device) for _ in range(3))
        mus, sigmas = [], []
        for k in range(N_PATCHES):  # estadisticos acumulados, como en el preentrenamiento
            (count, mu, sigma), _ = util.update_running_stats(count, mu, sigma, patches[:, k], mask[:, k])
            mus.append(mu)
            sigmas.append(sigma)
        mus, sigmas = torch.stack(mus, 1), torch.stack(sigmas, 1)
        self._buffer = {}
        self.m(util.revin(patches, mus, sigmas, reverse=False), mask)
        feats = []
        for i in self.idx:
            h = self._buffer[i]  # (N, parches, d_modelo)
            if self.pooling in ("mean", "meanlast"):
                feats.append(h.mean(1))
            if self.pooling in ("last", "meanlast"):
                feats.append(h[:, -1])
        if self.add_stats:
            feats.append(torch.stack([mus[:, -1], sigmas[:, -1]], dim=1))
        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def __call__(self, z):  # z: (B, C, T)
        z = torch.as_tensor(z, dtype=torch.float32)
        b, c, t = z.shape
        flat = z.reshape(b * c, t)
        parts = [self._series(flat[i:i + self.chunk].to(self.device)).cpu() for i in range(0, len(flat), self.chunk)]
        return torch.cat(parts).reshape(b, -1)


def _activate_norms(module):
    """Las normas RMS se inicializan en cero (salida nula) y dejarian los bloques como identidad; se fijan en uno."""
    for layer in module.modules():
        if hasattr(layer, "scale") and layer.__class__.__name__ == "RMSNorm":
            torch.nn.init.ones_(layer.scale)


def build_encoder(kind, layers=(-1,), pooling="mean", add_stats=True, checkpoint=None):
    """kind: 'timesfm' (pesos preentrenados), 'random' (misma arquitectura sin preentrenar) o 'identity'."""
    if kind == "identity":
        return IdentityEncoder()
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch as Wrapper
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module as Module

    if kind == "timesfm":
        module = Wrapper.from_pretrained(checkpoint or TIMESFM_REPO, torch_compile=False).model
    elif kind == "random":
        torch.manual_seed(0)
        module = Module()
        _activate_norms(module)
    else:
        raise ValueError(f"Codificador desconocido: {kind}")
    return TimesFMEncoder(module.to(DEVICE), layers=layers, pooling=pooling, add_stats=add_stats, name=kind)


@torch.no_grad()
def sanity_forecast_error(module, horizon=32, period=32, context=256):
    """Verifica la carga de pesos: error medio al pronosticar una onda seno (bajo si los pesos son los preentrenados)."""
    device = next(module.parameters()).device
    t = np.arange(context)
    x = torch.tensor(np.sin(2 * np.pi * t / period), dtype=torch.float32, device=device)[None]
    forecast, _, _ = module.decode(horizon, x, torch.zeros_like(x, dtype=torch.bool))
    pred = forecast[0, -1, :horizon, module.aridx].cpu().numpy()
    return float(np.mean(np.abs(pred - np.sin(2 * np.pi * (context + np.arange(horizon)) / period))))


class FeatureCache:
    """Caracteristicas estandarizadas con estadisticos del entrenamiento; las entradas fijas se codifican una vez."""

    def __init__(self, encoder, data):
        self.encoder, self._memo = encoder, {}
        feats = self.raw(data.Z["train"])
        self.mu, self.sd = feats.mean(0, keepdim=True), feats.std(0, keepdim=True).clamp_min(1e-6)

    def raw(self, z):  # ndarray (N, T, C) o tensor (B, C, T)
        if isinstance(z, np.ndarray):
            key = hashlib.md5(np.ascontiguousarray(z).tobytes()).hexdigest()
            if key not in self._memo:
                self._memo[key] = self.encoder(torch.from_numpy(z).permute(0, 2, 1))
            return self._memo[key]
        return self.encoder(z)

    def __call__(self, z):
        return ((self.raw(z) - self.mu) / self.sd).float()


class FoundationAutoencoder(nn.Module):
    """Proyeccion lineal entrenable de las caracteristicas a z y decodificador refinado."""

    def __init__(self, feat_dim, n_features, n_points, latent_dim=32, hidden_dim=48, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, latent_dim)
        self.decoder = GRUDecoder(latent_dim, n_features, n_points, hidden_dim, feed_latent=True)

    def forward(self, f):
        return self.decoder(self.head(self.drop(f))).permute(0, 2, 1)


def fit_foundation(data, cache, seed, latent_dim=32, closure_weight=0.1, denoising=False,
                   weight_decay=0.0, dropout=0.0, cfg: TrainConfig = TrainConfig()):
    """Entrena la proyeccion y el decodificador; con denoising las caracteristicas se recalculan en linea."""
    set_seed(seed)
    targets, feats = data.Z["train"], cache(data.Z["train"]).numpy()
    train = to_loader(targets, feats, batch_size=cfg.batch_size, shuffle=True)
    val = to_loader(data.Z["validation"], cache(data.Z["validation"]).numpy(), batch_size=cfg.batch_size, shuffle=False)
    model = FoundationAutoencoder(feats.shape[1], data.n_features, data.n_points, latent_dim, cfg.hidden_dim, dropout).to(DEVICE)

    def prepare(batch, training):
        target, f = batch[0].to(DEVICE), batch[1].to(DEVICE)
        if denoising and training:
            f = ((cache.encoder(corrupt_batch(target.cpu())) - cache.mu) / cache.sd).float().to(DEVICE)
        return f, target

    loss_fn = lambda p, t: reconstruction_loss(p, t, closure_weight)
    return fit_model(model, train, val, prepare, loss_fn, cfg, weight_decay=weight_decay)


@torch.no_grad()
def predict_foundation(model, cache, z):
    model.eval()
    return model(cache(z).to(DEVICE)).permute(0, 2, 1).cpu().numpy()


def run_foundation_seeds(data, cache, seeds=SEEDS, log=print, **kwargs):
    """Una red por semilla y el ensamble, con las mismas evaluaciones que los modelos GRU."""
    models, rows = [], []
    for seed in seeds:
        model = fit_foundation(data, cache, seed, **kwargs)
        row = evaluate(data, lambda z, m=model: predict_foundation(m, cache, z))
        row.update(seed=seed, epochs=model.n_epochs)
        rows.append(row)
        models.append(model)
        log(f"semilla {seed}: MAE {row['mae_deg']:.3f}  salto {row['endpoint_jump_deg']:.3f}  "
            f"correccion {row['correction_gain']:.3f}  tipicidad {row['typicality_gain']:.3f}  ({model.n_epochs} ep.)")
    ensemble = evaluate(data, lambda z: np.mean([predict_foundation(m, cache, z) for m in models], axis=0))
    ensemble["seed"] = "ensemble"
    return rows + [ensemble]
