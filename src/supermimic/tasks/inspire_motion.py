"""Motion tracking with optional hand labels on a floating-base Inspire G1."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
  resolve_motion_file,
)
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.utils.hand_tracking import (
  NUM_HAND_BODIES,
  fingertip_error_sq,
  fingertip_score,
)


def _load_hand_valid(path: Path, length: int) -> np.ndarray:
  if path.is_dir():
    mask_path = path / "hand_valid.npy"
    valid = np.load(mask_path) if mask_path.is_file() else np.ones(length, dtype=bool)
  else:
    with np.load(path) as data:
      valid = data.get("hand_valid", np.ones(length, dtype=bool))
  if valid.shape != (length,) or valid.dtype != np.bool_:
    raise ValueError(f"{path}: hand_valid must be a boolean array with {length} frames")
  return valid


class InspireMotionCommand(MotionCommand):
  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    if not cfg.motion_file:
      raise ValueError(
        "G1-Inspire requires motion files prepared with convert_inspire_motion"
      )
    super().__init__(cfg, env)
    if not np.isclose(self.motion.fps * env.step_dt, 1.0):
      raise ValueError(f"G1-Inspire expects reference FPS {1 / env.step_dt:g}")

    sources = [Path(path) for path in resolve_motion_file(cfg.motion_file)]
    lengths = self.motion.motion_ends.diff(prepend=self.motion.motion_ends.new_zeros(1))
    if len(sources) == 1 and sources[0].is_dir():
      lengths = lengths.sum().unsqueeze(0)
    valid = np.concatenate(
      [
        _load_hand_valid(path, length)
        for path, length in zip(sources, lengths.tolist(), strict=True)
      ]
    )
    # The same motion loader is used during data collection, BPTT and playback.
    self.hand_valid = torch.as_tensor(valid, device=self.device)
    self.motion.hand_valid = self.hand_valid  # pyright: ignore[reportAttributeAccessIssue]
    self.metrics["fingertip_rms_m"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["hand_reference_available"] = torch.zeros(
      self.num_envs, device=self.device
    )

  def hand_error_sq(self) -> torch.Tensor:
    return fingertip_error_sq(
      self.robot_body_pos_w,
      matrix_from_quat(self.robot_body_quat_w),
      self._body_pos_w,
      matrix_from_quat(self._body_quat_w),
    )

  def _update_metrics(self) -> None:
    super()._update_metrics()
    valid = self.hand_valid[self.time_steps]
    self.metrics["error_body_pos"] = (
      (
        self.body_pos_relative_w[:, :-NUM_HAND_BODIES]
        - self.robot_body_pos_w[:, :-NUM_HAND_BODIES]
      )
      .norm(dim=-1)
      .mean(dim=-1)
    )
    self.metrics["fingertip_rms_m"] = torch.where(
      valid, self.hand_error_sq().sqrt(), 0.0
    )
    self.metrics["hand_reference_available"] = valid.float()


@dataclass(kw_only=True)
class InspireMotionCommandCfg(MotionCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> InspireMotionCommand:
    return InspireMotionCommand(self, env)


def fingertip_tracking(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  command = env.command_manager.get_term("motion")
  assert isinstance(command, InspireMotionCommand)
  return fingertip_score(
    command.hand_error_sq(), command.hand_valid[command.time_steps], std
  )
