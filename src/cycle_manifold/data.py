"""Carga de ciclos normalizados, particion por sujeto y estandarizacion."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES, N_PHASE_POINTS, SPLITS

DEFAULT_RELATIVE_PATH = Path("data/cycles_normalized.parquet")
ENV_VARIABLE = "CYCLE_MANIFOLD_DATA"


@dataclass
class CycleData:
    """Ciclos en grados (X) y estandarizados con estadisticos de entrenamiento (Z)."""

    X: dict[str, np.ndarray]  # split -> (N, T, C)
    mean: np.ndarray  # (1, 1, C)
    std: np.ndarray  # (1, 1, C)
    features: tuple[str, ...] = FEATURES
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    Z: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.Z = {k: self.scale(v) for k, v in self.X.items()}

    @property
    def n_points(self) -> int:
        return self.X["train"].shape[1]

    @property
    def n_features(self) -> int:
        return self.X["train"].shape[2]

    def scale(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, z):
        return z * self.std + self.mean


def _from_parquet(path):
    frame = pd.read_parquet(path)
    ordered = frame.sort_values(["cycle_id", "phase_index"])
    n_cycles = ordered["cycle_id"].nunique()
    values = ordered[list(FEATURES)].to_numpy(np.float32).reshape(n_cycles, N_PHASE_POINTS, len(FEATURES))
    meta = ordered.drop_duplicates("cycle_id")[["cycle_id", "subject_id", "group", "anchor_side", "split"]]
    return values, meta.reset_index(drop=True)


def _from_npz(path):
    d = np.load(path, allow_pickle=False)
    meta = pd.DataFrame({"split": d["split"], "group": d["group"]})
    return d["X"].astype(np.float32), meta


def load_cycle_data(path) -> CycleData:
    """Lee cycles_normalized.parquet (o un .npz equivalente) y estandariza con el split train."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de ciclos: {path}")
    values, meta = _from_npz(path) if path.suffix == ".npz" else _from_parquet(path)
    split = meta["split"].to_numpy()
    X = {name: values[split == name] for name in SPLITS}
    mean = X["train"].mean(axis=(0, 1), keepdims=True)
    std = np.maximum(X["train"].std(axis=(0, 1), keepdims=True), 1e-4)
    return CycleData(X=X, mean=mean, std=std, meta=meta)


def find_cycles_file(path=None) -> Path:
    """Resuelve el archivo de ciclos: argumento, variable CYCLE_MANIFOLD_DATA o data/ en el repositorio."""
    import os

    if path:
        return Path(path)
    if os.environ.get(ENV_VARIABLE):
        return Path(os.environ[ENV_VARIABLE])
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / DEFAULT_RELATIVE_PATH).exists():
            return base / DEFAULT_RELATIVE_PATH
    raise FileNotFoundError(
        f"No se encontro {DEFAULT_RELATIVE_PATH}. Indique la ruta con --data o con la variable {ENV_VARIABLE} (vease data/README.md)"
    )
