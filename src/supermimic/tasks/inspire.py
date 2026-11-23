"""Unitree G1 with coupled Inspire DFQ finger joints."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityCfg

INSPIRE_TASK_ID = "SuperMimic-GenCoord-G1-Inspire"
HAND_SIDES = ("L", "R")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HAND_PALMS = tuple(f"{side}_hand_base_link" for side in HAND_SIDES)
FINGERTIPS = tuple(f"{side}_{finger}_tip" for side in HAND_SIDES for finger in FINGERS)
HAND_BODIES = (*HAND_PALMS, *FINGERTIPS)
INDEPENDENT_FINGER_JOINTS = tuple(
  f"{side}_{joint}_joint"
  for side in HAND_SIDES
  for joint in (
    "thumb_proximal_yaw",
    "thumb_proximal_pitch",
    "index_proximal",
    "middle_proximal",
    "ring_proximal",
    "pinky_proximal",
  )
)

_HAND_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "inspire"
# Nominal simulation PD gains. Calibrate against the target hand before deployment.
_HAND_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=INDEPENDENT_FINGER_JOINTS,
  stiffness=3.0,
  damping=0.05,
  effort_limit=1.0,
  armature=0.00002,
)
HAND_ACTION_SCALE = {
  name: 0.5 if "thumb_proximal_pitch" in name else 1.0
  for name in INDEPENDENT_FINGER_JOINTS
}


def get_g1_inspire_spec() -> mujoco.MjSpec:
  """Replace mjlab's fixed palms with the two Inspire hand branches."""
  spec = get_g1_robot_cfg().spec_fn()
  spec.copy_during_attach = True
  for side in ("left", "right"):
    wrist = spec.body(f"{side}_wrist_yaw_link")
    for geom in list(wrist.geoms):
      if (
        geom.meshname == f"{side}_rubber_hand" or geom.name == f"{side}_hand_collision"
      ):
        spec.delete(geom)
    spec.delete(spec.mesh(f"{side}_rubber_hand"))
    spec.delete(spec.site(f"{side}_palm"))

    hand = mujoco.MjSpec.from_file(str(_HAND_ASSETS / f"{side}_hand.xml"))
    mount = hand.body(f"{side}_inspire_mount")
    # Replace the wrist inertia that originally included the fixed rubber hand.
    wrist.mass = mount.mass
    wrist.ipos = mount.ipos
    xx, yy, zz, xy, xz, yz = mount.fullinertia
    principal = np.empty(3)
    rotation = np.empty(4)
    mujoco.mju_eig3(
      principal,
      np.empty(9),
      rotation,
      np.array((xx, xy, xz, xy, yy, yz, xz, yz, zz)),
    )
    wrist.inertia = principal
    wrist.iquat = rotation
    mount.to_frame()
    spec.attach(hand, prefix=f"inspire_{side}/", frame=wrist.add_frame())
  return spec


def get_g1_inspire_robot_cfg() -> EntityCfg:
  cfg = get_g1_robot_cfg()
  assert cfg.articulation is not None
  cfg.spec_fn = get_g1_inspire_spec
  cfg.articulation = replace(
    cfg.articulation, actuators=(*cfg.articulation.actuators, _HAND_ACTUATOR)
  )
  return cfg
