"""Metricas de forma, correccion y tipicidad (en grados salvo indicacion)."""
from __future__ import annotations

import numpy as np
import torch


def shape_metrics(data, target_z, pred_z):
    """Error y continuidad. Entradas estandarizadas (N, T, C); se reportan en grados."""
    t, p = data.inverse(target_z), data.inverse(pred_z)
    e = p - t
    return {
        "mae_deg": float(np.mean(np.abs(e))),
        "rmse_deg": float(np.sqrt(np.mean(e ** 2))),
        "endpoint_jump_deg": float(np.mean(np.abs(p[:, 0] - p[:, -1]))),
        "slope_jump_deg": float(np.mean(np.abs((p[:, 1] - p[:, 0]) - (p[:, -1] - p[:, -2])))),
        "vel_mae_deg": float(np.mean(np.abs(np.diff(p, axis=1) - np.diff(t, axis=1)))),
        "roughness_deg": float(np.mean(np.abs(np.diff(p, n=2, axis=1)))),
    }


def nearest_control_distance(train_z, x):
    """Distancia euclidea de cada ciclo aplanado al control de entrenamiento mas cercano."""
    a = torch.from_numpy(x.reshape(len(x), -1)).double()
    b = torch.from_numpy(train_z.reshape(len(train_z), -1)).double()
    return torch.cdist(a, b).min(dim=1).values.numpy()


def correction_gain(mae_output, mae_input):
    """1 - MAE(salida, limpio) / MAE(entrada danada, limpio)."""
    return 1.0 - mae_output / mae_input


def typicality_gain(dist_before, dist_after):
    """Media de 1 - d(salida) / d(entrada) sobre los casos."""
    return float(np.mean(1.0 - dist_after / dist_before))
