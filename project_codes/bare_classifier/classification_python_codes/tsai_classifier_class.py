from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from tsai.all import *  


def _get_backbone(name: str):
    """Resolve a tsai backbone class/callable by name from globals()."""
    bb = globals().get(name, None)
    if bb is None or not callable(bb):
        raise ValueError(f"Backbone model '{name}' not found or not callable.")
    return bb


class TSAIClassifierClass(nn.Module):
    """No-TDA classifier.

    All tsai backbones share the same interface: instantiate with
    (c_in, c_out, seq_len), accept [B, F, T], pool time internally, and
    return [B, c_out].

    Regularization note: the head normalizes the backbone output with
    LayerNorm before classification -- mirroring the TDA classifier head,
    whose first layer is LayerNorm(2*hidden_size). Without this, the
    un-normalized backbone output lets the head fit sample-specific
    feature magnitudes, which on small training sets yields a smooth
    train_loss but a noisy val_loss. The LayerNorm closes that gap
    without altering the backbone.
    """

    def __init__(self, args: Dict):
        super().__init__()
        self.args = args

        self.feature_dim   = int(args['feature_dim'])
        self.seq_len       = int(args['seq_len'])
        self.hidden_size   = int(args['hidden_size'])
        self.num_classes   = int(args['num_classes'])
        self.backbone_name = str(args['backbone_model'])

        # Optional knob; default preserves prior head dropout.
        head_dropout = float(args.get('head_dropout', 0.5))

        backbone_cls = _get_backbone(self.backbone_name)
        self.backbone = backbone_cls(
            c_in=self.feature_dim,
            c_out=self.hidden_size,
            seq_len=self.seq_len,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(self.hidden_size // 2, self.num_classes),
        )

        self._initialize_weights()

    def forward(self, time_series: torch.Tensor) -> torch.Tensor:
        x = time_series.float().transpose(1, 2)     # [B, F, T]
        feats = self.backbone(x)                    # [B, hidden_size]
        logits = self.head(feats)                   # [B, num_classes]
        return logits

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
