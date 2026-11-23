# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation
- `SuperMimic-GenCoord-G1-Inspire` — one policy for G1 body and both Inspire hands

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

### G1 with Inspire hands

The G1-Inspire task observes 53 joints and commands 41 independent joints. MuJoCo
constrains the remaining 12 finger joints to follow their drivers. Body tracking
and palm-relative fingertip tracking share the existing world-model training loop.
The included hand model uses Unitree kinematics and inertias with simplified visual
geometry. This task covers motion tracking in free space.

Download a 50 Hz reference archive from the [Weave dataset](https://huggingface.co/datasets/appolyn/Weave)
and convert a small subset to get started. Use the dataset's training split for
training and save test clips in a separate directory.

```bash
uv run python -m supermimic.scripts.convert_inspire_motion weave \
  WEAVE_TRAIN_ARCHIVE.npz --output-dir data/g1_inspire/train --limit 20

WANDB_MODE=disabled uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-G1-Inspire --motion-file 'data/g1_inspire/train/*.npz'

uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-G1-Inspire \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1-Inspire/RUN_DIRECTORY
```

To mix in existing 50 Hz G1 body clips, convert them to the same directory with
`convert_inspire_motion body BODY_CLIP.npz --output-dir data/g1_inspire/train`.
They keep their body references and carry a missing-hand label. Their finger
positions start neutral on reset and contribute no fingertip tracking gradient.
Keep the prepared clips as individual NPZ files so the hand labels travel with them.

The motion ONNX export includes the hand-reference mask and names the 41 action
channels separately from the 53 state joints. Use the local viewer for simulation
playback.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)
- [Unitree ROS](https://github.com/unitreerobotics/unitree_ros) (G1-Inspire hand
  kinematics and inertias, [BSD-3-Clause](src/supermimic/assets/inspire/LICENSE-Unitree.txt))

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
