"""Small supervised fusion heads over permanently frozen model features."""

from __future__ import annotations

import torch
from torch import nn


class FusionPredictor(nn.Module):
    def __init__(self, config: dict, assets: dict, feature_dims: list[int], genes: int):
        super().__init__()
        self.kind = config["model"]
        self.feature_dims = feature_dims
        width, dropout = config["width"], config["dropout"]
        self.target_projections = nn.ModuleList([nn.Linear(d, width) for d in feature_dims])
        self.context_projections = nn.ModuleList()
        for name in ["scgpt", "geneformer"]:
            context = torch.as_tensor(assets[f"{name}_context"])
            self.register_buffer(f"{name}_context", context)
            self.context_projections.append(nn.Linear(len(context), width))
        if self.kind in {"concat", "cross_attention"}:
            self.combine = nn.Sequential(nn.Linear(width * 2, width), nn.GELU())
        elif self.kind == "gated":
            self.gate = nn.Linear(width * 2, 1)
        else:
            raise ValueError(f"Unknown fusion: {self.kind}")
        if self.kind == "cross_attention":
            self.token_projections = nn.ModuleList()
            for name in ["scgpt", "geneformer"]:
                tokens = torch.as_tensor(assets[f"{name}_tokens"])
                ids = torch.as_tensor(assets[f"{name}_gene_indices"], dtype=torch.long)
                valid = torch.as_tensor(assets[f"{name}_valid"], dtype=torch.bool)
                if tokens.ndim != 2 or len(tokens) != len(ids) or len(valid) != len(ids):
                    raise ValueError("Gene tokens, identity and mask must have matching lengths")
                if not valid.any() or len(set(ids[valid].tolist())) != int(valid.sum()):
                    raise ValueError("Cross attention requires nonempty, uniquely identified genes")
                self.register_buffer(f"{name}_tokens", tokens)
                self.register_buffer(f"{name}_gene_indices", ids)
                self.register_buffer(f"{name}_valid", valid)
                self.token_projections.append(nn.Linear(tokens.shape[-1], width))
            self.cross = nn.ModuleList(
                [
                    nn.MultiheadAttention(width, config["heads"], dropout, batch_first=True)
                    for _ in range(2)
                ]
            )
            self.cross_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.head = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.GELU(),
        )
        self.decoder = nn.Linear(width, genes)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def branch_conditions(self, targets):
        split = targets.split(self.feature_dims, dim=-1)
        conditions = []
        for i, name in enumerate(["scgpt", "geneformer"]):
            context = getattr(self, f"{name}_context")
            context = torch.nn.functional.layer_norm(context, context.shape)
            value = self.context_projections[i](context) + self.target_projections[i](split[i])
            conditions.append(torch.nn.functional.gelu(value))
        return conditions

    def gate_weights(self, targets):
        if self.kind != "gated":
            raise ValueError("Only the gated head has gate weights")
        return torch.sigmoid(self.gate(torch.cat(self.branch_conditions(targets), -1)))

    def forward(self, targets):
        conditions = self.branch_conditions(targets)
        if self.kind == "gated":
            alpha = torch.sigmoid(self.gate(torch.cat(conditions, -1)))
            fused = alpha * conditions[0] + (1 - alpha) * conditions[1]
        elif self.kind == "concat":
            fused = self.combine(torch.cat(conditions, -1))
        else:
            sequences, masks = [], []
            for i, name in enumerate(["scgpt", "geneformer"]):
                valid = getattr(self, f"{name}_valid")
                # Mask before projection as well as in attention/pooling.
                tokens = getattr(self, f"{name}_tokens").masked_fill(~valid[:, None], 0)
                tokens = self.token_projections[i](tokens)
                tokens = tokens[None].expand(len(targets), -1, -1) + conditions[i][:, None]
                sequences.append(tokens)
                masks.append(valid)
            pooled = []
            for i in range(2):
                other = 1 - i
                attended, _ = self.cross[i](
                    sequences[i],
                    sequences[other],
                    sequences[other],
                    key_padding_mask=(~masks[other])[None].expand(len(targets), -1),
                    need_weights=False,
                )
                hidden = self.cross_norms[i](sequences[i] + attended)
                hidden = hidden.masked_fill(~masks[i][None, :, None], 0)
                pooled.append(hidden.sum(1) / masks[i].sum())
            fused = self.combine(torch.cat(pooled, -1))
        return self.decoder(self.head(fused))
