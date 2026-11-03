"""Las tres evaluaciones: fidelidad, correccion y tipicidad."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .config import CORRUPTION_SEED
from .corruption import corrupt_array
from .metrics import correction_gain, nearest_control_distance, shape_metrics, typicality_gain

METRICS = ["mae_deg", "endpoint_jump_deg", "slope_jump_deg", "correction_gain", "typicality_gain", "intervention_deg"]


def evaluate(data, predict_fn: Callable, corruption_seed=CORRUPTION_SEED):
    """predict_fn: Z (N, T, C) -> prediccion estandarizada. Devuelve las metricas de I, II y III."""
    z_test = data.Z["control_test"]
    out = shape_metrics(data, z_test, predict_fn(z_test))  # I: fidelidad
    perturbed = corrupt_array(z_test, corruption_seed)  # II: correccion
    base = shape_metrics(data, z_test, perturbed)["mae_deg"]
    out["correction_gain"] = correction_gain(shape_metrics(data, z_test, predict_fn(perturbed))["mae_deg"], base)
    z_case = data.Z["case_external_test"]  # III: tipicidad e intervencion
    d_before = nearest_control_distance(data.Z["train"], z_case)
    case_pred = predict_fn(z_case)
    out["typicality_gain"] = typicality_gain(d_before, nearest_control_distance(data.Z["train"], case_pred))
    out["intervention_deg"] = float(np.mean(np.abs(data.inverse(case_pred) - data.inverse(z_case))))
    return out
