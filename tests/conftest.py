import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cycle_manifold.data import CycleData  # noqa: E402


def make_synthetic(n_train=24, n_val=8, n_test=8, n_case=10, n_points=101, seed=0):
    """Ciclos periodicos sinteticos de seis canales con amplitud y fase variables."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_points)

    def block(n, scale):
        amp = rng.uniform(0.8, 1.2, size=(n, 1, 6)) * scale
        phase = rng.uniform(-0.1, 0.1, size=(n, 1, 6))
        return (amp * 20 * np.sin(2 * np.pi * (t[None, :, None] + phase))).astype(np.float32)

    X = {"train": block(n_train, 1.0), "validation": block(n_val, 1.0),
         "control_test": block(n_test, 1.0), "case_external_test": block(n_case, 1.5)}
    train = X["train"]
    return CycleData(X=X, mean=train.mean(axis=(0, 1), keepdims=True),
                     std=np.maximum(train.std(axis=(0, 1), keepdims=True), 1e-4))


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic()
