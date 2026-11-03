import numpy as np
import torch

from cycle_manifold.config import SPECS, TrainConfig
from cycle_manifold.corruption import corrupt_array, corrupt_batch
from cycle_manifold.losses import closure_loss, reconstruction_loss
from cycle_manifold.models import RNNAutoencoder
from cycle_manifold.metrics import correction_gain, shape_metrics, typicality_gain


def test_scaler_roundtrip(synthetic):
    z = synthetic.scale(synthetic.X["control_test"])
    assert np.allclose(synthetic.inverse(z), synthetic.X["control_test"], atol=1e-3)
    assert abs(synthetic.Z["train"].mean()) < 1e-3


def test_corrupt_batch_matches_loop_reference():
    x = torch.randn(16, 6, 101)

    def reference(x):  # version por ciclo con torch.roll
        center = x.mean(dim=2, keepdim=True)
        a = torch.empty((len(x), x.shape[1], 1)).uniform_(0.5, 1.8)
        b = torch.empty((len(x), x.shape[1], 1)).uniform_(-0.9, 0.9)
        r = center + a * (x - center) + b
        r = r + 0.15 * torch.randn_like(r)
        for i in range(len(r)):
            ls, rs = int(torch.randint(-8, 9, (1,))), int(torch.randint(-8, 9, (1,)))
            r[i, 0:3] = torch.roll(r[i, 0:3], ls, dims=1)
            r[i, 3:6] = torch.roll(r[i, 3:6], rs, dims=1)
        return r

    torch.manual_seed(5)
    expected = reference(x)
    torch.manual_seed(5)
    assert torch.equal(corrupt_batch(x), expected)


def test_corrupt_array_is_deterministic():
    x = np.random.default_rng(1).normal(size=(5, 101, 6)).astype(np.float32)
    assert np.array_equal(corrupt_array(x, 3), corrupt_array(x, 3))
    assert not np.array_equal(corrupt_array(x, 3), corrupt_array(x, 4))


def test_closure_loss_zero_for_periodic_signal():
    t = torch.linspace(0, 1, 101)
    wave = torch.sin(2 * np.pi * t)
    wave = wave.expand(2, 6, 101).clone()  # sin(0) = sin(2 pi)
    assert closure_loss(wave) < 1e-5
    open_wave = torch.linspace(0, 1, 101).expand(2, 6, 101)
    assert abs(closure_loss(open_wave).item() - 1.0) < 1e-6


def test_reconstruction_loss_weight():
    p, t = torch.zeros(2, 6, 101), torch.zeros(2, 6, 101)
    p[:, :, 0] = 1.0
    base = reconstruction_loss(p, t, 0.0)
    assert reconstruction_loss(p, t, 0.5) > base


def test_model_shapes_and_decoder_input():
    for feed in (False, True):
        model = RNNAutoencoder(6, 101, latent_dim=8, hidden_dim=12, feed_latent=feed)
        x = torch.randn(4, 6, 101)
        assert model(x).shape == x.shape
        assert model.encode(x).shape == (4, 8)
        assert model.decoder.rnn.input_size == 2 + (8 if feed else 0)


def test_specs_follow_protocol():
    assert SPECS["A_refined"].latent_dim == 32 and SPECS["A_refined"].closure_weight == 0.1
    assert SPECS["B_refined"].denoising and SPECS["B_refined"].closure_weight == 0.05
    assert not SPECS["A_original"].feed_latent and SPECS["A_original"].closure_weight == 0.0


def test_metrics_known_values(synthetic):
    z = synthetic.Z["control_test"]
    perfect = shape_metrics(synthetic, z, z)
    assert perfect["mae_deg"] == 0.0 and perfect["rmse_deg"] == 0.0
    shifted = shape_metrics(synthetic, z, z + 1.0 / synthetic.std)  # +1 grado en todos los canales
    assert abs(shifted["mae_deg"] - 1.0) < 1e-3
    assert abs(correction_gain(2.0, 4.0) - 0.5) < 1e-12
    assert abs(typicality_gain(np.array([2.0, 4.0]), np.array([1.0, 2.0])) - 0.5) < 1e-12
