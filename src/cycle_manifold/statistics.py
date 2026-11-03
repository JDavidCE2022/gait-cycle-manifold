"""Resumen por semillas y pruebas pareadas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .evaluation import METRICS


def split_rows(rows):
    individual = [r for r in rows if r["seed"] != "ensemble"]
    ensemble = next(r for r in rows if r["seed"] == "ensemble")
    return individual, ensemble


def summarize(named_rows, metrics=METRICS):
    """Media y desviacion (ddof=1) por semilla, mas el valor del ensamble."""
    table = []
    for name, rows in named_rows.items():
        ind, ens = split_rows(rows)
        entry = {"model": name}
        for m in metrics:
            values = np.array([r[m] for r in ind])
            std = values.std(ddof=1) if len(values) > 1 else np.nan
            entry[m + "_mean"], entry[m + "_std"], entry[m + "_ensemble"] = values.mean(), std, ens[m]
        table.append(entry)
    return pd.DataFrame(table).set_index("model")


def paired_test(rows, reference_rows, metrics=METRICS):
    """Prueba t pareada por semilla (alfa 0.05, sin correccion por multiplicidad)."""
    a = {r["seed"]: r for r in rows if r["seed"] != "ensemble"}
    b = {r["seed"]: r for r in reference_rows if r["seed"] != "ensemble"}
    seeds = [s for s in a if s in b]
    out = []
    for m in metrics:
        x, y = np.array([a[s][m] for s in seeds]), np.array([b[s][m] for s in seeds])
        t, p = stats.ttest_rel(x, y)
        out.append({"metric": m, "model": x.mean(), "reference": y.mean(), "difference": (x - y).mean(), "p_value": p})
    return pd.DataFrame(out).set_index("metric")
