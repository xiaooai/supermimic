"""Hand references and fingertip scoring shared by simulation and imagination."""

import torch

FINGERS_PER_HAND = 5
NUM_HAND_BODIES = 2 + 2 * FINGERS_PER_HAND
FINGERTIP_REWARD = "fingertip_tracking"


def fingertip_error_sq(
  pos: torch.Tensor,
  rot: torch.Tensor,
  target_pos: torch.Tensor,
  target_rot: torch.Tensor,
) -> torch.Tensor:
  """Mean squared palm-frame fingertip distance, in square meters.

  The tracked bodies end with the two palms, then five tips per hand.
  """

  def palm_frame(points: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    palms = points[..., -NUM_HAND_BODIES : -2 * FINGERS_PER_HAND, :]
    tips = points[..., -2 * FINGERS_PER_HAND :, :].unflatten(-2, (2, FINGERS_PER_HAND))
    palm_rot = rotations[..., -NUM_HAND_BODIES : -2 * FINGERS_PER_HAND, :, :]
    return (tips - palms.unsqueeze(-2)) @ palm_rot

  error = palm_frame(pos, rot) - palm_frame(target_pos, target_rot)
  return error.square().sum(dim=-1).mean(dim=(-1, -2))


def fingertip_score(
  error_sq: torch.Tensor, hand_valid: torch.Tensor, std: float
) -> torch.Tensor:
  # A constant on unlabelled frames keeps the return-based reset ceiling uniform.
  return torch.where(hand_valid, torch.exp(-error_sq / std**2), 1.0)
