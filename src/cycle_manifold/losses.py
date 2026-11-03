"""Perdidas de reconstruccion y de cierre del ciclo."""
from __future__ import annotations

import torch


def closure_loss(prediction):
    """Promedio de |G(0) - G(100)| sobre lote y canales. prediction: (B, C, T)."""
    return torch.mean(torch.abs(prediction[:, :, 0] - prediction[:, :, -1]))


def reconstruction_loss(prediction, target, closure_weight=0.0):
    """MSE estandarizado mas w veces la perdida de cierre."""
    loss = torch.mean((prediction - target) ** 2)
    if closure_weight:
        loss = loss + closure_weight * closure_loss(prediction)
    return loss
