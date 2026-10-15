"""Randomly initialized encoders with a shared all-gene residual decoder."""

from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, tokens: int, features: int, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tokens + features, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.GELU(),
        )

    def forward(self, target, expression, gene_features):
        return self.net(torch.cat([expression.expand(len(target), -1), target], dim=-1))


class TransformerEncoder(nn.Module):
    def __init__(self, features: int, width: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.gene_projection = nn.Linear(features + 1, width)
        self.target_projection = nn.Linear(features, width)
        self.target_type = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, target, expression, gene_features):
        # A gene set has no artificial positional order. Identity comes from descriptors.
        context = self.gene_projection(torch.cat([gene_features, expression[:, None]], -1))
        context = context[None].expand(len(target), -1, -1)
        condition = self.target_projection(target)[:, None] + self.target_type
        sequence = torch.cat([condition, context], dim=1)
        return self.norm(self.encoder(sequence)[:, 0])


class ResidualPredictor(nn.Module):
    def __init__(
        self, config: dict, descriptors: torch.Tensor, expression: torch.Tensor, genes: int
    ):
        super().__init__()
        self.register_buffer("context_features", descriptors)
        self.register_buffer("context_expression", expression)
        features = descriptors.shape[-1]
        width = config["width"]
        if config["model"] == "mlp":
            self.encoder = MLPEncoder(len(expression), features, width, config["dropout"])
        elif config["model"] == "transformer":
            self.encoder = TransformerEncoder(
                features, width, config["layers"], config["heads"], config["dropout"]
            )
        else:
            raise ValueError(f"Unknown encoder: {config['model']}")
        self.decoder = nn.Linear(width, genes)
        # Start at identity; the encoder remains randomly initialized.
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        return self.decoder(
            self.encoder(target_features, self.context_expression, self.context_features)
        )


class FrozenFeaturePredictor(nn.Module):
    """Frozen foundation-model features -> trainable head -> shared mean shift.

    Feature extraction happens once outside supervised training. Averaging the
    control-cell embeddings preserves the existing condition-only architecture.
    """

    def __init__(self, config: dict, context: torch.Tensor, feature_dim: int, genes: int):
        super().__init__()
        width = config["width"]
        self.register_buffer("context", context)
        self.target_projection = nn.Linear(feature_dim, width)
        self.context_projection = nn.Linear(len(context), width)
        self.encoder = nn.Sequential(
            nn.Linear(width * 2, width * 4),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(width * 4, width),
            nn.GELU(),
        )
        self.decoder = nn.Linear(width, genes)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        context = torch.nn.functional.layer_norm(self.context, self.context.shape)
        context = self.context_projection(context).expand(len(target_features), -1)
        target = self.target_projection(target_features)
        return self.decoder(self.encoder(torch.cat([context, target], dim=-1)))


def apply_residual(controls: torch.Tensor, delta: torch.Tensor, target_sum: float) -> torch.Tensor:
    """Unpaired control bank + condition shift, clipped and library-renormalized.

    Shape: controls [C,G], delta [P,G], result [P,C,G].
    The output is continuous log1p-normalized expression, never rounded counts.
    """
    log_expression = torch.clamp(controls[None] + delta[:, None], min=0, max=14)
    expression = torch.expm1(log_expression)
    expression = expression * (target_sum / expression.sum(-1, keepdim=True).clamp_min(1e-6))
    return torch.log1p(expression)
