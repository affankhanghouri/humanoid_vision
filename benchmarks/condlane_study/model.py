"""Inference-only CondLaneNet Small graph reconstructed from the official source.

The module names intentionally match the official checkpoint.  The exported
graph ends at the fixed-size heatmap, dynamic parameters, and mask features;
seed selection and instance decoding remain ordinary NumPy postprocessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


INPUT_WIDTH = 800
INPUT_HEIGHT = 320
CULANE_WIDTH = 1640
CULANE_HEIGHT = 590
CULANE_CROP_TOP = 270
MEAN = np.asarray([75.3, 76.6, 77.6], dtype=np.float32)
STD = np.asarray([50.5, 53.8, 54.3], dtype=np.float32)


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int,
                 padding: int = 0, norm: bool = False, relu: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, padding=padding, bias=not norm
        )
        if norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv(value)
        if hasattr(self, "bn"):
            value = self.bn(value)
        if hasattr(self, "relu"):
            value = self.relu(value)
        return value


def position_embedding(channels: int, height: int, width: int) -> torch.Tensor:
    mask = torch.zeros((1, height, width), dtype=torch.bool)
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    features = channels // 2
    dim_t = torch.arange(features, dtype=torch.float32)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / features)
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), 4).flatten(3)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), 4).flatten(3)
    return torch.cat((pos_y, pos_x), 3).permute(0, 3, 1, 2)


class AttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ratio: int = 4):
        super().__init__()
        self.pre_conv = ConvModule(in_dim, out_dim, 3, padding=1, norm=True, relu=True)
        self.query_conv = nn.Conv2d(out_dim, out_dim // ratio, 1)
        self.key_conv = nn.Conv2d(out_dim, out_dim // ratio, 1)
        self.value_conv = nn.Conv2d(out_dim, out_dim, 1)
        self.final_conv = ConvModule(out_dim, out_dim, 3, padding=1, norm=True, relu=True)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, value: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        value = self.pre_conv(value) + pos
        batch, _, height, width = value.shape
        query = self.query_conv(value).view(batch, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(value).view(batch, -1, width * height)
        attention = self.softmax(torch.bmm(query, key)).permute(0, 2, 1)
        projected = self.value_conv(value).view(batch, -1, width * height)
        output = torch.bmm(projected, attention).view(batch, -1, height, width)
        return self.final_conv(self.gamma * output + value)


class TransHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_layers = nn.ModuleList([
            AttentionLayer(512, 64),
            AttentionLayer(64, 64),
        ])
        self.register_buffer("pos_0", position_embedding(64, 10, 25), persistent=False)
        self.register_buffer("pos_1", position_embedding(64, 10, 25), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.attn_layers[0](value, self.pos_0)
        return self.attn_layers[1](value, self.pos_1)


class TransConvFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_head = TransHead()
        self.lateral_convs = nn.ModuleList([
            ConvModule(128, 64, 1),
            ConvModule(256, 64, 1),
            ConvModule(64, 64, 1),
        ])
        self.fpn_convs = nn.ModuleList([
            ConvModule(64, 64, 3, padding=1),
            ConvModule(64, 64, 3, padding=1),
            ConvModule(64, 64, 3, padding=1),
        ])

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        transformed = self.trans_head(features[-1])
        inputs = (features[1], features[2], transformed)
        laterals = [layer(value) for layer, value in zip(self.lateral_convs, inputs)]
        for index in range(2, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index], size=laterals[index - 1].shape[2:], mode="nearest"
            )
        return tuple(layer(value) for layer, value in zip(self.fpn_convs, laterals))


class CtnetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hm = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
        self.params = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 134, 1))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hm(value), self.params(value)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv1d(100, 64, 1), nn.Conv1d(64, 2, 1)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers[1](F.relu(self.layers[0](value)))


class CondLaneHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_branch = nn.Sequential(
            ConvModule(64, 64, 3, padding=1, norm=True),
            ConvModule(64, 64, 3, padding=1, norm=True),
            ConvModule(64, 64, 3, padding=1, norm=True, relu=True),
        )
        self.ctnet_head = CtnetHead()
        self.mlp = MLP()

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        heatmap, params = self.ctnet_head(features[1])
        mask_features = self.mask_branch(features[0])
        return heatmap, params, mask_features


class CondLaneDense(nn.Module):
    """Fixed graph corresponding to official CondLaneNet inference stage 1."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.neck = TransConvFPN()
        self.bbox_head = CondLaneHead()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        backbone = self.backbone
        value = backbone.relu(backbone.bn1(backbone.conv1(image)))
        value = backbone.maxpool(value)
        layer1 = backbone.layer1(value)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)
        return self.bbox_head(self.neck((layer1, layer2, layer3, layer4)))


def load_model(checkpoint: Path) -> CondLaneDense:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    model = CondLaneDense()
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"backbone.fc.weight", "backbone.fc.bias"}
    # The released checkpoint retains an obsolete reg_branch. The official
    # inference source aliases reg_branch = mask_branch and never runs it.
    allowed_unexpected = all(key.startswith("bbox_head.reg_branch.") for key in unexpected)
    if set(missing) != allowed_missing or not allowed_unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


@dataclass(frozen=True)
class Transform:
    original_width: int
    original_height: int
    crop_top: int


def preprocess(image: np.ndarray) -> tuple[np.ndarray, Transform]:
    height, width = image.shape[:2]
    crop_top = round(height * CULANE_CROP_TOP / CULANE_HEIGHT)
    cropped = image[crop_top:, :]
    resized = cv2.resize(cropped, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    tensor = ((resized.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), Transform(width, height, crop_top)


def _local_peaks(heatmap: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(heatmap[0, 0], -30, 30)))
    pooled = cv2.dilate(probability, np.ones((3, 3), np.uint8))
    ys, xs = np.where((probability >= threshold) & (probability == pooled))
    seeds = [(int(y), int(x), float(probability[y, x])) for y, x in zip(ys, xs)]
    groups: list[list[tuple[int, int, float]]] = []
    for seed in seeds:
        for group in groups:
            if any(math.hypot(seed[1] - old[1], seed[0] - old[0]) <= 4 for old in group):
                group.append(seed)
                break
        else:
            groups.append([seed])
    return [max(group, key=lambda item: item[2]) for group in groups]


def decode_lanes(outputs: tuple[np.ndarray, ...], transform: Transform,
                 threshold: float = 0.5) -> list[dict[str, object]]:
    heatmap, parameters, mask_features = outputs
    locations_y, locations_x = np.mgrid[0:40, 0:100].astype(np.float32)
    # Preserve the official implementation: both coordinate planes are divided by width.
    features = np.concatenate([
        (locations_x / 100.0)[None, None],
        (locations_y / 100.0)[None, None],
        mask_features,
    ], axis=1)[0]
    lanes = []
    for y_seed, x_seed, score in _local_peaks(heatmap, threshold):
        dynamic = parameters[0, :, y_seed, x_seed]
        mask = np.tensordot(dynamic[:66], features, axes=(0, 0)) + dynamic[66] - 2.19
        regression = np.tensordot(dynamic[67:133], features, axes=(0, 0)) + dynamic[133]
        # The range MLP is applied by the runtime-specific helper after dense inference.
        lanes.append({"score": score, "mask": mask, "regression": regression})
    return lanes


def finish_decode(raw_lanes: list[dict[str, object]], mlp_weights: tuple[np.ndarray, ...],
                  transform: Transform) -> list[dict[str, object]]:
    w0, b0, w1, b1 = mlp_weights
    result = []
    for raw in raw_lanes:
        mask = np.asarray(raw["mask"], dtype=np.float32)
        shifted = mask - mask.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        columns = (probs * np.arange(100, dtype=np.float32)).sum(axis=1)
        # Conv1d with kernel 1: channels are the 100 mask columns, positions are rows.
        hidden = np.maximum(0, w0[:, :, 0] @ mask.T + b0[:, None])
        ranges = w1[:, :, 0] @ hidden + b1[:, None]
        valid = np.argmax(ranges, axis=0).astype(bool)
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            continue
        first = max(0, int(indices[0]) - 1)
        last = min(39, int(indices[-1]) + 1)
        rows = np.arange(first, last + 1)
        discrete = columns.astype(np.int32).clip(0, 99)
        offsets = np.asarray(raw["regression"])[rows, discrete[rows]]
        model_x = (discrete[rows].astype(np.float32) + offsets) * 8.0
        model_y = rows.astype(np.float32) * 8.0
        x = model_x * transform.original_width / INPUT_WIDTH
        y = transform.crop_top + model_y * (transform.original_height - transform.crop_top) / INPUT_HEIGHT
        points = [[float(px), float(py)] for px, py in zip(x, y)]
        if len(points) >= 2:
            result.append({"score": float(raw["score"]), "points": points})
    return result


def mlp_weights(model: CondLaneDense) -> tuple[np.ndarray, ...]:
    layers = model.bbox_head.mlp.layers
    return tuple(value.detach().cpu().numpy() for value in (
        layers[0].weight, layers[0].bias, layers[1].weight, layers[1].bias
    ))
