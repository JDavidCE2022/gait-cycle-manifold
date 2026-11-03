"""Autoencoder recurrente (GRU) del manifold de ciclos."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def phase_encoding(n_points):
    """Codificacion periodica de la fase: (sin 2 pi t, cos 2 pi t), forma (T, 2)."""
    t = torch.linspace(0.0, 1.0, n_points)
    return torch.stack([torch.sin(2 * np.pi * t), torch.cos(2 * np.pi * t)], dim=-1)


class GRUDecoder(nn.Module):
    """Decodificador GRU: z fija el estado inicial y, opcionalmente, entra en cada paso."""

    def __init__(self, latent_dim, n_features, n_points, hidden_dim, feed_latent):
        super().__init__()
        self.n_points, self.feed_latent = n_points, feed_latent
        self.init = nn.Linear(latent_dim, hidden_dim)
        self.rnn = nn.GRU(2 + (latent_dim if feed_latent else 0), hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, n_features)
        self.register_buffer("phase", phase_encoding(n_points))

    def forward(self, z):
        h0 = self.init(z).unsqueeze(0)
        inputs = self.phase.unsqueeze(0).expand(len(z), -1, -1)
        if self.feed_latent:
            inputs = torch.cat([inputs, z.unsqueeze(1).expand(-1, self.n_points, -1)], dim=-1)
        decoded, _ = self.rnn(inputs, h0)
        return self.out(decoded)


class RNNAutoencoder(nn.Module):
    """Codificador GRU bidireccional y decodificador GRU. Entrada y salida: (B, C, T)."""

    def __init__(self, n_features, n_points, latent_dim=16, hidden_dim=48, feed_latent=False):
        super().__init__()
        self.encoder_rnn = nn.GRU(n_features, hidden_dim, batch_first=True, bidirectional=True)
        self.encoder_fc = nn.Linear(2 * hidden_dim, latent_dim)
        self.decoder = GRUDecoder(latent_dim, n_features, n_points, hidden_dim, feed_latent)

    def encode(self, x):
        _, h = self.encoder_rnn(x.permute(0, 2, 1))
        return self.encoder_fc(torch.cat([h[0], h[1]], dim=-1))

    def forward(self, x):
        return self.decoder(self.encode(x)).permute(0, 2, 1)
