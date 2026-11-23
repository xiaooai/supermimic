"""Regression coverage for coupled finger states and optional hand targets."""

from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity

from supermimic.representations import NUM_GOAL_FRAMES, motion_goal_window
from supermimic.scripts.convert_inspire_motion import (
  WEAVE_JOINTS,
  InspireMotionConverter,
)
from supermimic.tasks.inspire import FINGERTIPS, get_g1_inspire_robot_cfg
from supermimic.utils.hand_tracking import fingertip_error_sq, fingertip_score


def test_inspire_model_has_coupled_state_and_independent_action_layout():
  robot = Entity(get_g1_inspire_robot_cfg())
  model = robot.spec.compile()
  assert (len(robot.joint_names), model.nu, model.neq) == (53, 41, 12)
  _, actions = robot.find_joints_by_actuator_names((".*",))
  assert "L_thumb_proximal_pitch_joint" in actions
  assert "L_thumb_distal_joint" in robot.joint_names
  assert "L_thumb_distal_joint" not in actions
  assert set(FINGERTIPS) <= set(robot.body_names)
  assert (model.eq_type == mujoco.mjtEq.mjEQ_JOINT).sum() == 12


def test_motion_conversion_aligns_names_couples_fingers_and_marks_missing_hands(
  tmp_path,
):
  converter = InspireMotionConverter()
  n = 3
  clip = {
    "root_pos_w": np.zeros((n, 3)),
    "root_quat_w": np.tile((1, 0, 0, 0), (n, 1)),
    "root_lin_vel_w": np.zeros((n, 3)),
    "root_ang_vel_w": np.zeros((n, 3)),
    "joint_pos": np.zeros((n, 53)),
    "joint_vel": np.zeros((n, 53)),
  }
  driver = "L_thumb_proximal_pitch_joint"
  follower = "L_thumb_distal_joint"
  clip["joint_pos"][:, WEAVE_JOINTS.index(driver)] = (0.25, 0.3, 0.8)
  clip["joint_vel"][:, WEAVE_JOINTS.index(driver)] = 0.25
  converter.write_clip(
    clip, WEAVE_JOINTS, tmp_path / "hand.npz", has_hands=True, fps=50
  )
  body_clip = {
    **clip,
    "joint_pos": clip["joint_pos"][:, :29],
    "joint_vel": clip["joint_vel"][:, :29],
  }
  converter.write_clip(
    body_clip, WEAVE_JOINTS[:29], tmp_path / "body.npz", has_hands=False, fps=50
  )

  with np.load(tmp_path / "hand.npz") as hand, np.load(tmp_path / "body.npz") as body:
    names = tuple(hand["joint_names"])
    np.testing.assert_allclose(
      hand["joint_pos"][:, names.index(follower)], [0.6, 0.72, 1.2]
    )
    np.testing.assert_allclose(
      hand["joint_vel"][:, names.index(follower)], [0.6, 0.6, 0.0]
    )
    np.testing.assert_allclose(body["joint_pos"][:, names.index(driver)], 0.0)
    motion = SimpleNamespace(
      joint_pos=torch.from_numpy(
        np.concatenate((hand["joint_pos"], body["joint_pos"]))
      ),
      joint_vel=torch.from_numpy(
        np.concatenate((hand["joint_vel"], body["joint_vel"]))
      ),
      hand_valid=torch.from_numpy(
        np.concatenate((hand["hand_valid"], body["hand_valid"]))
      ),
      motion_ends=torch.tensor([n, 2 * n]),
    )
  goal = motion_goal_window(motion, torch.tensor([n - 1, n])).reshape(
    2, NUM_GOAL_FRAMES, 107
  )
  assert torch.all(goal[0, :, -1] == 1)  # Future frames stop at the clip boundary.
  assert torch.all(goal[1, :, -1] == 0)


def test_fingertip_reward_is_palm_relative_and_gates_the_gradient():
  pos = torch.zeros((2, 12, 3))
  rot = torch.eye(3).expand(2, 12, 3, 3)
  target = pos + torch.tensor([2.0, 0.0, 1.0])
  target[:, -1, 2] += 0.03
  pos.requires_grad_()
  error = fingertip_error_sq(pos, rot, target, rot)
  torch.testing.assert_close(error, torch.full((2,), 0.03**2 / 10))
  score = fingertip_score(error, torch.tensor([True, False]), std=0.025)
  score.sum().backward()
  assert pos.grad is not None
  assert pos.grad[0].abs().sum() > 0
  assert torch.all(pos.grad[1] == 0)
