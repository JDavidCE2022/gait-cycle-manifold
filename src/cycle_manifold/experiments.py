"""Protocolo de cinco semillas con ensamble para los modelos GRU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import SEEDS, SPECS, ModelSpec, TrainConfig
from .evaluation import evaluate
from .statistics import paired_test, summarize
from .training import fit_gru, predict_gru


def run_gru_seeds(data, spec: ModelSpec, seeds=SEEDS, cfg: TrainConfig = TrainConfig(), log=print):
    """Una red por semilla y el ensamble (promedio de predicciones)."""
    models, rows = [], []
    for seed in seeds:
        model = fit_gru(data, spec, seed, cfg)
        row = evaluate(data, lambda z, m=model: predict_gru(m, z))
        row.update(seed=seed, epochs=model.n_epochs)
        rows.append(row)
        models.append(model)
        log(f"{spec.name} semilla {seed}: MAE {row['mae_deg']:.3f}  salto {row['endpoint_jump_deg']:.3f}  "
            f"correccion {row['correction_gain']:.3f}  tipicidad {row['typicality_gain']:.3f}  ({model.n_epochs} ep.)")
    ensemble = evaluate(data, lambda z: np.mean([predict_gru(m, z) for m in models], axis=0))
    ensemble["seed"] = "ensemble"
    log(f"{spec.name} ensamble: MAE {ensemble['mae_deg']:.3f}  salto {ensemble['endpoint_jump_deg']:.3f}  "
        f"correccion {ensemble['correction_gain']:.3f}")
    return rows + [ensemble]


def run_gru_benchmark(data, names=tuple(SPECS), seeds=SEEDS, cfg: TrainConfig = TrainConfig(), out_dir=None, log=print):
    """Entrena y evalua las variantes pedidas; guarda JSON, resumen y pruebas pareadas."""
    results = {name: run_gru_seeds(data, SPECS[name], seeds, cfg, log) for name in names}
    summary = summarize(results)
    tests = {}
    for kind in ("A", "B"):  # refinado frente a original, por semilla
        if f"{kind}_refined" in results and f"{kind}_original" in results:
            tests[kind] = paired_test(results[f"{kind}_refined"], results[f"{kind}_original"])
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gru_per_seed.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        summary.to_csv(out / "gru_summary.csv")
        for kind, table in tests.items():
            table.to_csv(out / f"gru_paired_{kind}_refined_vs_original.csv")
    return results, summary, tests
