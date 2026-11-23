"""Prepare Weave or existing body-only clips for the SuperMimic G1-Inspire task."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import Entity
from mjlab.tasks.tracking.mdp.commands import MOTION_FIELDS, load_clip

from supermimic.tasks.inspire import INDEPENDENT_FINGER_JOINTS, get_g1_inspire_robot_cfg

# Isaac Lab articulation order from the Weave data card:
# https://huggingface.co/datasets/appolyn/Weave#joint-and-body-ordering
WEAVE_JOINTS = tuple(
  """
left_hip_pitch_joint right_hip_pitch_joint waist_yaw_joint
left_hip_roll_joint right_hip_roll_joint waist_roll_joint
left_hip_yaw_joint right_hip_yaw_joint waist_pitch_joint left_knee_joint
right_knee_joint left_shoulder_pitch_joint right_shoulder_pitch_joint
left_ankle_pitch_joint right_ankle_pitch_joint left_shoulder_roll_joint
right_shoulder_roll_joint left_ankle_roll_joint right_ankle_roll_joint
left_shoulder_yaw_joint right_shoulder_yaw_joint left_elbow_joint
right_elbow_joint left_wrist_roll_joint right_wrist_roll_joint
left_wrist_pitch_joint right_wrist_pitch_joint left_wrist_yaw_joint
right_wrist_yaw_joint L_index_proximal_joint L_middle_proximal_joint
L_pinky_proximal_joint L_ring_proximal_joint L_thumb_proximal_yaw_joint
R_index_proximal_joint R_middle_proximal_joint R_pinky_proximal_joint
R_ring_proximal_joint R_thumb_proximal_yaw_joint L_index_intermediate_joint
L_middle_intermediate_joint L_pinky_intermediate_joint
L_ring_intermediate_joint L_thumb_proximal_pitch_joint
R_index_intermediate_joint R_middle_intermediate_joint
R_pinky_intermediate_joint R_ring_intermediate_joint
R_thumb_proximal_pitch_joint L_thumb_intermediate_joint
R_thumb_intermediate_joint L_thumb_distal_joint R_thumb_distal_joint
""".split()
)

WEAVE_ROOT_FIELDS = {
  "root_pos_w": "body_pos_w",
  "root_quat_w": "body_quat_w",
  "root_lin_vel_w": "body_lin_vel_w",
  "root_ang_vel_w": "body_ang_vel_w",
}
MOTION_FPS = 50


class InspireMotionConverter:
  def __init__(self) -> None:
    robot = Entity(get_g1_inspire_robot_cfg())
    self.joint_names = robot.joint_names
    self.joint_columns = {name: i for i, name in enumerate(self.joint_names)}

    model = robot.spec.compile()
    model_joints = {
      model.joint(j).name.rsplit("/", 1)[-1]: model.joint(j) for j in range(model.njnt)
    }
    self.drive_limits = {
      self.joint_columns[name]: model_joints[name].range.copy()
      for name in INDEPENDENT_FINGER_JOINTS
    }
    self.couplings = []
    for eq in range(model.neq):
      if model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
        continue
      follower = model.joint(int(model.eq_obj1id[eq])).name.rsplit("/", 1)[-1]
      driver = model.joint(int(model.eq_obj2id[eq])).name.rsplit("/", 1)[-1]
      if driver in INDEPENDENT_FINGER_JOINTS:
        offset, ratio = model.eq_data[eq, :2]
        self.couplings.append(
          (self.joint_columns[follower], self.joint_columns[driver], ratio, offset)
        )

  def write_clip(
    self,
    clip: dict[str, np.ndarray],
    joint_names: tuple[str, ...],
    dst: Path,
    *,
    has_hands: bool,
    fps: float,
  ) -> None:
    if not np.isclose(fps, MOTION_FPS):
      raise ValueError(f"{dst.name}: expected {MOTION_FPS} fps, got {fps:g}")
    expected = (
      set(self.joint_names)
      if has_hands
      else {name for name in self.joint_names if not name.startswith(("L_", "R_"))}
    )
    if set(joint_names) != expected or len(joint_names) != len(expected):
      missing, extra = (
        sorted(expected - set(joint_names)),
        sorted(set(joint_names) - expected),
      )
      raise ValueError(
        f"{dst.name}: joints mismatch (missing {missing}, extra {extra})"
      )

    n_frames = len(clip["joint_pos"])
    if n_frames < 2:
      raise ValueError(f"{dst.name}: a clip needs at least two frames")
    columns = [self.joint_columns[name] for name in joint_names]
    output = {}
    for key, width in zip(
      MOTION_FIELDS, (3, 4, 3, 3, len(columns), len(columns)), strict=True
    ):
      values = np.asarray(clip[key], dtype=np.float32)
      if values.shape != (n_frames, width) or not np.isfinite(values).all():
        raise ValueError(
          f"{dst.name}: {key} must contain finite values of shape {(n_frames, width)}"
        )
      if key.startswith("joint_"):
        expanded = np.zeros((n_frames, len(self.joint_names)), dtype=np.float32)
        expanded[:, columns] = values
        values = expanded
      output[key] = values

    pos, vel = output["joint_pos"], output["joint_vel"]
    for driver, (lower, upper) in self.drive_limits.items():
      clamped = (pos[:, driver] < lower) | (pos[:, driver] > upper)
      pos[:, driver] = np.clip(pos[:, driver], lower, upper)
      vel[clamped, driver] = 0.0
    for follower, driver, ratio, offset in self.couplings:
      pos[:, follower] = offset + ratio * pos[:, driver]
      vel[:, follower] = ratio * vel[:, driver]

    with dst.open("xb") as file:
      np.savez_compressed(
        file,
        **output,
        joint_names=np.asarray(self.joint_names),
        fps=np.asarray(MOTION_FPS),
        hand_valid=np.full(n_frames, has_hands, dtype=bool),
      )

  def convert_weave(self, source: Path, output_dir: Path, limit: int | None) -> int:
    with np.load(source) as data:
      lengths = data["motion_lengths"]
      fps = float(data["fps"].item())
      motion = {
        key: data[source_key][:, 0].copy()
        for key, source_key in WEAVE_ROOT_FIELDS.items()
      }
      motion.update(joint_pos=data["joint_pos"], joint_vel=data["joint_vel"])
    if (
      lengths.ndim != 1
      or not np.issubdtype(lengths.dtype, np.integer)
      or ((lengths < 2).any() or lengths.sum() != len(motion["joint_pos"]))
    ):
      raise ValueError(
        f"{source}: motion_lengths must cover all frames with clips of length >= 2"
      )

    start = 0
    for index, length in enumerate(lengths[:limit]):
      end = start + int(length)
      self.write_clip(
        {key: values[start:end] for key, values in motion.items()},
        WEAVE_JOINTS,
        output_dir / f"{source.stem}_{index:04d}.npz",
        has_hands=True,
        fps=fps,
      )
      start = end
    return len(lengths[:limit])

  def convert_body(self, source: Path, output_dir: Path) -> None:
    clip = load_clip(str(source))
    self.write_clip(
      clip,
      tuple(clip["joint_names"].tolist()),
      output_dir / f"{source.stem}_body.npz",
      has_hands=False,
      fps=float(clip["fps"].item()),
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("format", choices=("weave", "body"))
  parser.add_argument("sources", type=Path, nargs="+")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--limit", type=int, help="Maximum number of clips per Weave archive"
  )
  args = parser.parse_args()
  if args.limit is not None and (args.limit < 1 or args.format != "weave"):
    parser.error("--limit requires Weave input and a positive count")

  args.output_dir.mkdir(parents=True, exist_ok=True)
  converter = InspireMotionConverter()
  count = 0
  for source in args.sources:
    if args.format == "weave":
      count += converter.convert_weave(source, args.output_dir, args.limit)
    else:
      converter.convert_body(source, args.output_dir)
      count += 1
  print(f"Wrote {count} clip(s) to {args.output_dir}")


if __name__ == "__main__":
  main()
