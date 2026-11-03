import numpy as np
import pandas as pd
import pytest

from cycle_manifold.config import FEATURES, N_PHASE_POINTS
from cycle_manifold.data import find_cycles_file, load_cycle_data


def _write_parquet(path, n_per_split=3):
    rng = np.random.default_rng(0)
    rows = []
    for s, split in enumerate(("train", "validation", "control_test", "case_external_test")):
        for k in range(n_per_split):
            cid = f"c{s}{k}"
            frame = pd.DataFrame(rng.normal(size=(N_PHASE_POINTS, 6)), columns=FEATURES)
            frame.insert(0, "phase_index", np.arange(N_PHASE_POINTS))
            frame.insert(0, "cycle_id", cid)
            frame["subject_id"], frame["group"] = f"s{s}", "control"
            frame["anchor_side"], frame["split"] = "left", split
            rows.append(frame)
    pd.concat(rows).sample(frac=1.0, random_state=0).to_parquet(path)  # filas desordenadas


def test_load_parquet_orders_and_splits(tmp_path):
    path = tmp_path / "cycles_normalized.parquet"
    _write_parquet(path)
    data = load_cycle_data(path)
    assert data.X["train"].shape == (3, N_PHASE_POINTS, 6)
    assert set(data.X) == {"train", "validation", "control_test", "case_external_test"}
    assert np.allclose(data.mean, data.X["train"].mean(axis=(0, 1), keepdims=True))


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cycle_data(tmp_path / "no_existe.parquet")


def test_real_project_counts_when_available():
    try:
        path = find_cycles_file()
    except FileNotFoundError:
        pytest.skip("datos del proyecto no disponibles")
    data = load_cycle_data(path)
    assert {k: len(v) for k, v in data.X.items()} == {
        "train": 240, "validation": 44, "control_test": 47, "case_external_test": 130}
