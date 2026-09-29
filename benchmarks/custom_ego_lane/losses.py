"""Safety-weighted, visibility-masked losses for structured ego lanes."""
import torch
import torch.nn.functional as F

def ego_lane_loss(predictions, coordinates, visible, state, sample_weight=None, geometry_weight=None):
    predicted_x, visibility_logits, state_logits = predictions
    weights = torch.ones_like(state, dtype=predicted_x.dtype) if sample_weight is None else sample_weight
    geometry = torch.ones_like(weights) if geometry_weight is None else geometry_weight
    state_ce = F.cross_entropy(state_logits, state, reduction="none")
    state_scale = torch.where(state == 1, 3.0, 1.0).to(weights.dtype)
    state_loss = (state_ce * state_scale * weights).sum() / weights.sum().clamp_min(1.0)
    vis_raw = F.binary_cross_entropy_with_logits(visibility_logits, visible, reduction="none").mean((1, 2))
    vis_weight = geometry * weights
    visibility_loss = (vis_raw * state_scale * vis_weight).sum() / vis_weight.sum().clamp_min(1.0)
    coordinate_raw = F.smooth_l1_loss(predicted_x, coordinates, reduction="none", beta=0.02)
    mask = visible * weights[:, None, None] * geometry[:, None, None]
    coordinate_loss = (coordinate_raw * mask).sum() / mask.sum().clamp_min(1.0)
    total = state_loss + 1.5 * visibility_loss + 4.0 * coordinate_loss
    return {"total": total, "state": state_loss, "visibility": visibility_loss, "coordinate": coordinate_loss}
