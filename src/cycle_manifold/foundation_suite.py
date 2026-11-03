"""Experimento completo del codificador fundacional: seleccion de configuracion, controles y tablas."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from .config import SEEDS, TrainConfig
from .foundation import (FeatureCache, build_encoder, fit_foundation, predict_foundation, run_foundation_seeds,
                         sanity_forecast_error)
from .metrics import shape_metrics

GRID = list(itertools.product((-1, 9), ("mean", "meanlast"), (0.0, 1e-3)))  # bloque, agrupacion, decaimiento
TYPE_A = dict(latent_dim=32, closure_weight=0.1, denoising=False)
TYPE_B = dict(latent_dim=16, closure_weight=0.05, denoising=True)


def explore_configs(data, encoder, seed=42, cfg: TrainConfig = TrainConfig(), log=print):
    """Ocho configuraciones evaluadas sobre validacion con una semilla (sin usar el test)."""
    rows = []
    for layer, pooling, wd in GRID:
        encoder.configure([layer], pooling)
        cache = FeatureCache(encoder, data)
        model = fit_foundation(data, cache, seed, weight_decay=wd, cfg=cfg, **TYPE_A)
        row = shape_metrics(data, data.Z["validation"], predict_foundation(model, cache, data.Z["validation"]))
        row.update(bloque=layer, agrupacion=pooling, decaimiento=wd, dim=int(cache.mu.shape[1]))
        rows.append(row)
        log(f"bloque {layer:>2} {pooling:<8} wd {wd:g}: MAE validacion {row['mae_deg']:.3f}  dim {row['dim']}")
    return sorted(rows, key=lambda r: r["mae_deg"])


def _tag(rows, kind, name):
    return [{"tipo": kind, "modelo": name, **r} for r in rows]


def run_suite(data, checkpoint=None, dry_run=False, seeds=SEEDS, cfg: TrainConfig = TrainConfig(),
              out_dir=None, log=print):
    """Reproduce la tabla del codificador fundacional: TimesFM, TimesFM aleatorio y serie cruda (tipos A y B).

    dry_run reemplaza los pesos preentrenados por pesos aleatorios: valida el flujo sin descargar el modelo.
    """
    pretrained = build_encoder("random" if dry_run else "timesfm", checkpoint=checkpoint)
    log(f"verificacion de pesos (error de pronostico de una onda seno): {sanity_forecast_error(pretrained.m):.3f}")
    exploration = explore_configs(data, pretrained, cfg=cfg, log=log)
    best = exploration[0]
    chosen = dict(layer=best["bloque"], pooling=best["agrupacion"], wd=best["decaimiento"], exploracion=exploration)
    log(f"configuracion elegida: bloque {chosen['layer']}, {chosen['pooling']}, wd {chosen['wd']:g}")
    pretrained.configure([chosen["layer"]], chosen["pooling"])
    rows = []

    def run(kind, name, encoder, params, wd):
        log(f"{kind}: {name}")
        out = run_foundation_seeds(data, FeatureCache(encoder, data), seeds, weight_decay=wd, cfg=cfg, log=log, **params)
        rows.extend(_tag(out, kind, name))

    run("A", "TimesFM preentrenado", pretrained, TYPE_A, chosen["wd"])
    random_enc = build_encoder("random")
    random_enc.configure([chosen["layer"]], chosen["pooling"])
    run("A", "TimesFM aleatorio", random_enc, TYPE_A, chosen["wd"])
    run("A", "Serie cruda", build_encoder("identity"), TYPE_A, 0.0)
    run("B", "TimesFM preentrenado", pretrained, TYPE_B, chosen["wd"])
    run("B", "Serie cruda", build_encoder("identity"), TYPE_B, 0.0)

    table = pd.DataFrame(rows)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "resultados_modelo_fundacional.csv", index=False)
        (out / "configuracion_elegida.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    return table, chosen
