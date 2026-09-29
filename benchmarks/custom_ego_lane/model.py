"""Tiny structured ego-lane network; no production dependency."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


INPUT_SIZE = (192, 320)
ROW_ANCHORS = 24


class TinyEgoLaneNet(nn.Module):
    """Predict two x coordinates/visibilities per row plus a 3-state ego head."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.row_projection = nn.Sequential(
            nn.Conv2d(576, 48, 1, bias=False), nn.BatchNorm2d(48), nn.Hardswish(),
        )
        self.row_head = nn.Linear(48 * 10, 4)  # left x, right x, left vis logit, right vis logit
        self.state_head = nn.Sequential(nn.Linear(576, 64), nn.Hardswish(), nn.Dropout(0.1), nn.Linear(64, 3))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.features(image)
        rows = F.interpolate(self.row_projection(feature), size=(ROW_ANCHORS, 10), mode="bilinear", align_corners=False).transpose(1, 2).flatten(2)
        row_output = self.row_head(rows)
        coordinates = torch.sigmoid(row_output[..., :2])
        visibility_logits = row_output[..., 2:]
        state_logits = self.state_head(feature.mean(dim=(2, 3)))
        return coordinates, visibility_logits, state_logits


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
