"""Constantes y configuraciones del protocolo experimental."""
from __future__ import annotations

from dataclasses import dataclass

FEATURES = ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")
N_PHASE_POINTS = 101
SPLITS = ("train", "validation", "control_test", "case_external_test")
SEEDS = (42, 1, 7, 123, 2024)
CORRUPTION_SEED = 42  # realizacion fija del dano para la evaluacion II


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparametros de entrenamiento compartidos por todos los modelos."""

    batch_size: int = 64
    epochs: int = 100
    patience: int = 12
    lr: float = 1e-3
    clip_norm: float = 1.0
    hidden_dim: int = 48
    min_delta: float = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    """Variante del autoencoder GRU.

    feed_latent: entrega z al decodificador en cada paso (mejora 1).
    closure_weight: peso w de la perdida de cierre |G(0) - G(100)| (mejora 2).
    denoising: entrena con entrada corrupta y objetivo limpio (modelo B).
    """

    name: str
    latent_dim: int
    feed_latent: bool
    closure_weight: float
    denoising: bool


SPECS = {
    "A_original": ModelSpec("A_original", 16, False, 0.0, False),
    "B_original": ModelSpec("B_original", 16, False, 0.0, True),
    "A_refined": ModelSpec("A_refined", 32, True, 0.1, False),
    "B_refined": ModelSpec("B_refined", 16, True, 0.05, True),
}
