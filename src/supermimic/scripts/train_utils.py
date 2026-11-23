"""Shared utilities for training scripts."""

from __future__ import annotations

from pathlib import Path

import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.mdp.commands import resolve_motion_file
from mjlab.utils.torch import configure_torch_backends

from supermimic.utils.wandb_artifacts import download_artifact_cached


def resolve_local_motion_files(spec: str) -> list[str]:
  """Expand every path or glob in a comma-separated local motion argument."""
  return [
    str(Path(path).resolve())
    for source in spec.split(",")
    for path in resolve_motion_file(source.strip())
  ]


def download_registry_motions(registry_name: str | None) -> str | list[str] | None:
  """Download motion files from W&B registry artifacts.

  Supports comma-separated registry names for multi-motion.
  Returns a single path string if one motion, a list if multiple, or None.
  """
  if not registry_name:
    return None

  names = [n.strip() for n in registry_name.split(",")]
  api = wandb.Api()

  paths = []
  for name in names:
    if ":" not in name:
      name = name + ":latest"
    artifact = api.artifact(name)
    paths.append(str(download_artifact_cached(artifact) / "motion.npz"))

  print(f"[INFO] Loaded {len(paths)} motion(s)")
  return paths if len(paths) > 1 else paths[0]


def setup_device(device: str | None = None) -> str:
  """Configure torch backends and resolve device."""
  configure_torch_backends()
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def create_env(
  env_cfg: ManagerBasedRlEnvCfg, num_envs: int, device: str
) -> RslRlVecEnvWrapper:
  """Set ``env_cfg.scene.num_envs`` and build a wrapped RL environment."""
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env)
