import numpy as np

from cycle_manifold.config import SPECS, TrainConfig
from cycle_manifold.evaluation import METRICS, evaluate
from cycle_manifold.experiments import run_gru_benchmark
from cycle_manifold.foundation import FeatureCache, IdentityEncoder, fit_foundation, predict_foundation
from cycle_manifold.statistics import paired_test, summarize
from cycle_manifold.training import fit_gru, predict_gru

FAST = TrainConfig(epochs=3, patience=2, hidden_dim=12)


def test_fit_gru_is_reproducible(synthetic):
    a = fit_gru(synthetic, SPECS["A_refined"], 1, FAST)
    b = fit_gru(synthetic, SPECS["A_refined"], 1, FAST)
    pa, pb = predict_gru(a, synthetic.Z["control_test"]), predict_gru(b, synthetic.Z["control_test"])
    assert np.allclose(pa, pb, atol=1e-6)


def test_evaluate_returns_all_metrics(synthetic):
    model = fit_gru(synthetic, SPECS["B_refined"], 1, FAST)
    result = evaluate(synthetic, lambda z: predict_gru(model, z))
    assert set(METRICS) <= set(result)
    assert all(np.isfinite(v) for v in result.values())


def test_benchmark_and_statistics(synthetic, tmp_path):
    results, summary, tests = run_gru_benchmark(
        synthetic, names=("A_original", "A_refined"), seeds=(1, 2, 3), cfg=FAST, out_dir=tmp_path, log=lambda *_: None)
    assert (tmp_path / "gru_summary.csv").exists() and (tmp_path / "gru_per_seed.json").exists()
    assert results["A_refined"][-1]["seed"] == "ensemble" and len(results["A_refined"]) == 4
    assert set(summary.index) == {"A_original", "A_refined"}
    assert "A" in tests and "p_value" in tests["A"].columns


def test_paired_test_detects_constant_shift():
    base = [{"seed": s, "mae_deg": 5.0 + 0.1 * s, **{m: 0.0 for m in ("endpoint_jump_deg", "slope_jump_deg", "correction_gain", "typicality_gain", "intervention_deg")}} for s in range(5)]
    better = [{**r, "mae_deg": r["mae_deg"] - 1.0 + 0.01 * i} for i, r in enumerate(base)]
    table = paired_test(better, base, metrics=["mae_deg"])
    assert table.loc["mae_deg", "difference"] < -0.9 and table.loc["mae_deg", "p_value"] < 0.001


def test_foundation_head_with_identity_encoder(synthetic):
    cache = FeatureCache(IdentityEncoder(), synthetic)
    model = fit_foundation(synthetic, cache, 1, latent_dim=8, closure_weight=0.1, cfg=FAST)
    pred = predict_foundation(model, cache, synthetic.Z["control_test"])
    assert pred.shape == synthetic.Z["control_test"].shape
    denoised = fit_foundation(synthetic, cache, 1, latent_dim=8, denoising=True, cfg=FAST)
    assert np.isfinite(predict_foundation(denoised, cache, synthetic.Z["case_external_test"])).all()


def test_random_timesfm_blocks_are_active():
    import pytest
    pytest.importorskip("timesfm")
    import torch
    from cycle_manifold.foundation import build_encoder

    encoder = build_encoder("random")
    x = torch.randn(2, 6, 101)
    encoder.configure([0], "mean")
    shallow = encoder(x)
    encoder.configure([9], "mean")
    deep = encoder(x)
    assert (shallow - deep).abs().max() > 1e-3  # con normas en cero ambos serian identicos
