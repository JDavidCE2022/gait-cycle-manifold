"""Bucle de entrenamiento con parada temprana, compartido por todos los modelos."""
from __future__ import annotations

import copy
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import ModelSpec, TrainConfig
from .corruption import corrupt_batch
from .losses import reconstruction_loss
from .models import RNNAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_loader(*arrays, batch_size, shuffle):
    """Arrays (N, T, C) se entregan como (B, C, T); los vectores 2D quedan igual."""
    tensors = [torch.from_numpy(a).permute(0, 2, 1) if a.ndim == 3 else torch.from_numpy(a) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def fit_model(model, train_loader, val_loader, prepare: Callable, loss_fn: Callable, cfg: TrainConfig,
              weight_decay=0.0):
    """Entrena con Adam y recorte de gradiente; conserva el mejor estado en validacion.

    prepare(batch, training) -> (entrada, objetivo), ambos ya en el dispositivo.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)
    best_state, best_val, wait, epoch = None, np.inf, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            inputs, target = prepare(batch, True)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(inputs), target).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, target = prepare(batch, False)
                total += loss_fn(model(inputs), target).item() * len(target)
                count += len(target)
        val = total / count
        if val < best_val - cfg.min_delta:
            best_val, wait, best_state = val, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.n_epochs, model.best_val = epoch, best_val
    return model


def fit_gru(data, spec: ModelSpec, seed, cfg: TrainConfig = TrainConfig()):
    """Entrena un autoencoder GRU (original o refinado, limpio o denoising)."""
    set_seed(seed)
    train = to_loader(data.Z["train"], batch_size=cfg.batch_size, shuffle=True)
    val = to_loader(data.Z["validation"], batch_size=cfg.batch_size, shuffle=False)
    model = RNNAutoencoder(data.n_features, data.n_points, spec.latent_dim, cfg.hidden_dim, spec.feed_latent).to(DEVICE)

    def prepare(batch, training):
        target = batch[0].to(DEVICE)
        return (corrupt_batch(target) if (spec.denoising and training) else target), target

    loss_fn = lambda p, t: reconstruction_loss(p, t, spec.closure_weight)
    return fit_model(model, train, val, prepare, loss_fn, cfg)


@torch.no_grad()
def predict_gru(model, z):
    """z: (N, T, C) estandarizado -> prediccion (N, T, C) estandarizada."""
    model.eval()
    x = torch.from_numpy(z).permute(0, 2, 1).to(DEVICE)
    return model(x).permute(0, 2, 1).cpu().numpy()
