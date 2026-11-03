"""Corrupcion sintetica de ciclos (modelo B y evaluacion II)."""
from __future__ import annotations

import numpy as np
import torch

from .config import CORRUPTION_SEED

AMPLITUDE_RANGE = (0.5, 1.8)
OFFSET_RANGE = (-0.9, 0.9)
NOISE_STD = 0.15
MAX_SHIFT = 8


def _roll_sides(x, shifts, time_dim):
    """Desplazamiento circular por ciclo: lado izquierdo (canales 0:3) y derecho (3:6)."""
    n = x.shape[time_dim]
    grid = torch.arange(n, device=x.device)
    out = x.clone()
    for side, sl in enumerate((slice(0, 3), slice(3, 6))):
        idx = (grid[None, :] - shifts[:, side, None]) % n  # (B, T), equivale a torch.roll
        part = x[:, sl] if time_dim == 2 else x[:, :, sl]
        if time_dim == 2:
            out[:, sl] = torch.gather(part, 2, idx[:, None, :].expand_as(part))
        else:
            out[:, :, sl] = torch.gather(part, 1, idx[:, :, None].expand_as(part))
    return out


def corrupt_batch(x):
    """x: (B, C, T) estandarizado. Usa el generador global de torch."""
    center = x.mean(dim=2, keepdim=True)
    shape = (len(x), x.shape[1], 1)
    amplitude = torch.empty(shape, device=x.device).uniform_(*AMPLITUDE_RANGE)
    offset = torch.empty(shape, device=x.device).uniform_(*OFFSET_RANGE)
    result = center + amplitude * (x - center) + offset
    result = result + NOISE_STD * torch.randn_like(result)
    shifts = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (len(x), 2), device=x.device)
    return _roll_sides(result, shifts, time_dim=2)


def corrupt_array(x, seed=CORRUPTION_SEED):
    """x: (N, T, C). Realizacion determinista (generador de numpy) para evaluar."""
    rng = np.random.default_rng(seed)
    center = x.mean(axis=1, keepdims=True)
    amplitude = rng.uniform(*AMPLITUDE_RANGE, size=(len(x), 1, x.shape[2]))
    offset = rng.uniform(*OFFSET_RANGE, size=(len(x), 1, x.shape[2]))
    result = center + amplitude * (x - center) + offset
    result += rng.normal(0, NOISE_STD, size=result.shape)
    for i in range(len(result)):  # dos sorteos por ciclo, en orden fijo
        result[i, :, :3] = np.roll(result[i, :, :3], rng.integers(-MAX_SHIFT, MAX_SHIFT + 1), axis=0)
        result[i, :, 3:] = np.roll(result[i, :, 3:], rng.integers(-MAX_SHIFT, MAX_SHIFT + 1), axis=0)
    return result.astype(np.float32)
