"""Linea de comandos: python -m cycle_manifold {gru,foundation,foundation-suite,export}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import SEEDS, SPECS
from .data import find_cycles_file, load_cycle_data
from .metrics import shape_metrics


def _seeds(text):
    return tuple(int(s) for s in text.split(","))


def _load(args):
    return load_cycle_data(find_cycles_file(args.data))


def cmd_gru(args):
    from .experiments import run_gru_benchmark

    data = _load(args)
    _, summary, tests = run_gru_benchmark(data, args.specs, args.seeds, out_dir=args.out)
    compact = [col for col in summary.columns if col.startswith(("mae", "endpoint", "correction", "typicality"))]
    print(summary[compact].round(3).to_string())
    for kind, table in tests.items():
        print(f"\n{kind}: refinado frente a original (prueba t pareada)")
        print(table.round(4).to_string())


def cmd_foundation(args):
    from .foundation import FeatureCache, build_encoder, fit_foundation, predict_foundation, run_foundation_seeds
    from .statistics import summarize

    data = _load(args)
    encoder = build_encoder(args.encoder, [int(x) for x in args.layers.split(",")], args.pooling, not args.no_stats, args.checkpoint)
    cache = FeatureCache(encoder, data)
    print("dimension de caracteristicas:", cache.mu.shape[1])
    kw = dict(latent_dim=args.latent, closure_weight=args.w, denoising=args.denoising,
              weight_decay=args.weight_decay, dropout=args.dropout)
    if args.split == "validation":  # exploracion de configuracion sin tocar el test
        rows = []
        for seed in args.seeds:
            model = fit_foundation(data, cache, seed, **kw)
            row = shape_metrics(data, data.Z["validation"], predict_foundation(model, cache, data.Z["validation"]))
            row.update(seed=seed, epochs=model.n_epochs, val_loss=model.best_val)
            rows.append(row)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
    else:
        rows = run_foundation_seeds(data, cache, args.seeds, **kw)
        print(summarize({args.encoder: rows}).round(3).T.to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


def cmd_suite(args):
    from .config import TrainConfig
    from .foundation_suite import run_suite

    data = _load(args)
    cfg = TrainConfig(epochs=args.epochs) if args.epochs else TrainConfig()
    table, _ = run_suite(data, args.checkpoint, args.dry_run, args.seeds, cfg, args.out)
    ind = table[table["seed"].astype(str) != "ensemble"]
    cols = ["mae_deg", "endpoint_jump_deg", "correction_gain", "typicality_gain", "intervention_deg"]
    print(ind.groupby(["tipo", "modelo"])[cols].agg(["mean", "std"]).round(3).to_string())


def cmd_export(args):
    data = _load(args)
    split = np.concatenate([[name] * len(x) for name, x in data.X.items()])
    X = np.concatenate(list(data.X.values()))
    np.savez_compressed(args.out, X=X, split=split, group=np.where(split == "case_external_test", "case", "control"),
                        features=np.array(data.features))
    print("guardado", args.out, X.shape)


def build_parser():
    p = argparse.ArgumentParser(prog="cycle_manifold", description=__doc__)
    p.add_argument("--data", help="cycles_normalized.parquet o .npz (por defecto data/cycles_normalized.parquet)")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gru", help="GRU original frente a refinado, con cinco semillas")
    g.add_argument("--specs", nargs="+", default=list(SPECS), choices=list(SPECS))
    g.add_argument("--seeds", type=_seeds, default=SEEDS)
    g.add_argument("--out", help="directorio de salida")
    g.set_defaults(func=cmd_gru)

    f = sub.add_parser("foundation", help="codificador congelado y decodificador GRU")
    f.add_argument("--encoder", default="timesfm", choices=["timesfm", "random", "identity"])
    f.add_argument("--checkpoint", help="ruta local de los pesos de TimesFM")
    f.add_argument("--layers", default="9", help="bloque(s) del transformador (configuracion elegida: 9)")
    f.add_argument("--pooling", default="meanlast", choices=["mean", "last", "meanlast"])
    f.add_argument("--no-stats", action="store_true")
    f.add_argument("--latent", type=int, default=32)
    f.add_argument("--w", type=float, default=0.1)
    f.add_argument("--weight-decay", type=float, default=1e-3)
    f.add_argument("--dropout", type=float, default=0.0)
    f.add_argument("--denoising", action="store_true")
    f.add_argument("--seeds", type=_seeds, default=SEEDS)
    f.add_argument("--split", default="test", choices=["validation", "test"])
    f.add_argument("--out", help="archivo JSON de salida")
    f.set_defaults(func=cmd_foundation)

    s = sub.add_parser("foundation-suite", help="seleccion de configuracion, controles y tablas completas")
    s.add_argument("--checkpoint", help="ruta local de los pesos de TimesFM")
    s.add_argument("--dry-run", action="store_true", help="pesos aleatorios: valida el flujo sin descargar el modelo")
    s.add_argument("--seeds", type=_seeds, default=SEEDS)
    s.add_argument("--epochs", type=int, help="limita las epocas (solo para pruebas rapidas)")
    s.add_argument("--out", default="results/foundation")
    s.set_defaults(func=cmd_suite)

    e = sub.add_parser("export", help="exporta los ciclos a un .npz compacto (p. ej. para Colab)")
    e.add_argument("--out", default="cycles_array.npz")
    e.set_defaults(func=cmd_export)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as error:
        raise SystemExit(f"Error: {error}") from None


if __name__ == "__main__":
    main()
