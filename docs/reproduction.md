# Training and reproduction

[Back to README](../README.md)

## What this checkout provides

The source includes the main learning components and stage-specific scripts for Sokoban and FrozenLake. It does not currently bundle prepared datasets, trained checkpoints, or the exact evaluation manifests. Public entry points for Navigation, PrimitiveSkill, and EB-ALFRED are pending.

This guide maps the existing implementation. It is not a claim that the shell defaults reproduce every result in the manuscript.

## Primary entry points

The full-pipeline wrappers are [`run_sokoban_full_training.sh`](../scripts/sokoban/run_sokoban_full_training.sh) and [`run_frozenlake_full_training.sh`](../scripts/frozenlake/run_frozenlake_full_training.sh). They accept `PHASE=data|base_wm|prior|wm|reward|critic|ppo|rl|all` and orchestrate the shared stages below. Set `PYTHON_BIN` if your interpreter is not in `.venv/bin/python`.

Both wrappers default to a frozen world model. Set `WORLD_MODEL_MODE=alternating_wm` explicitly to enable refresh. The Sokoban wrapper defaults to 50,000 offline episodes; the FrozenLake wrapper defaults to 5,000. Both use 400 accepted actor updates by default. Confirm the paper experiment overrides before comparison.

## Stage dependencies

| Stage | Script | Required inputs / outputs |
| --- | --- | --- |
| Collection | [`collect_offline_dataset.py`](../scripts/train/collect_offline_dataset.py) | Produces raw RGB episodes and a manifest |
| Paired features | [`build_qwen_vjepa_teacher_dataset.py`](../scripts/train/build_qwen_vjepa_teacher_dataset.py) | Raw replay + Qwen + V-JEPA; produces paired tokens |
| World model | [`run_qwen_vjepa_teacher_wm.sh`](../scripts/train/run_qwen_vjepa_teacher_wm.sh) | `PHASE=prepare`, `train`, `audit`, or `all`; needs local Qwen and replay |
| Prior diagnostics / repair | [`run_qwen_vjepa_prior_repair.sh`](../scripts/train/run_qwen_vjepa_prior_repair.sh) | Development stages referenced by downstream defaults |
| Reward head | [`run_qwen_vjepa_reward_head.sh`](../scripts/train/run_qwen_vjepa_reward_head.sh) | Gated WM, semantic gate report, paired replay; produces reward checkpoint |
| Critic warm-up | [`run_qwen_vjepa_critic_warmup.sh`](../scripts/train/run_qwen_vjepa_critic_warmup.sh) | Reward checkpoint, paired replay, H2 cache; produces critic and warm-up report |
| PPO | [`run_qwen_vjepa_ppo.sh`](../scripts/train/run_qwen_vjepa_ppo.sh) | Critic, reward checkpoint, reports, replay, fixed evaluation file |
| Alternating WM + PPO | [`run_qwen_vjepa_ppo_alternating_wm.sh`](../scripts/train/run_qwen_vjepa_ppo_alternating_wm.sh) | Wraps PPO with real-replay world-model refresh |
| FrozenLake | [`run_frozenlake_training.sh`](../scripts/frozenlake/run_frozenlake_training.sh) | Environment-specific orchestration; also expects external evaluation resources |

## Configure paths explicitly

Review `MODEL_PATH`, `PYTHON_BIN`, `DATA_DIR`, `SOURCE_WM`, `STAGE1_GATE_REPORT`, `REWARD_CKPT`, `CRITIC_CKPT`, `CRITIC_REPORT`, and the stage output directories. These variables have stage-specific meanings: do not assume each script’s defaults point to the output of the immediately preceding script.

For example, the reward runner’s default source is a prior-repair checkpoint rather than the initial world-model checkpoint. The critic warm-up creates an H2 cache, and the PPO preflight requires both checkpoint files and validation reports.

The FrozenLake wrapper also references `VAGEN_ROOT`. Provide the exact compatible resources rather than reusing the original machine-specific path.

## Preflight without launching PPO

Once all stage artifacts exist, the PPO runner supports `PREFLIGHT_ONLY=1`. Set the paths in your shell first:

```bash
PREFLIGHT_ONLY=1 bash scripts/train/run_qwen_vjepa_ppo.sh
```

This validates prerequisites; it does not create missing data or checkpoints and is not an end-to-end performance test.

## Match an experiment before comparing results

Record the following with each run:

- Code revision and full configuration, including overrides to the shell defaults.
- Raw data provenance, split manifests, action mapping, and evaluation levels/seeds.
- Backbone and teacher versions; world-model, reward, and critic initialization.
- Imagined rollout mode and horizon; whether counterfactual H2 training is enabled.
- Accepted and rejected actor updates, newly collected real transitions, and compute accounting.
- World-model refresh policy and its validation guard; reward calibration thresholds.

The current alternating runner defaults to `ROLLOUT_HORIZON=2`; the PPO runner explicitly enables `COUNTERFACTUAL_H2_PPO=1`. These need reconciliation with the manuscript’s sampled-rollout description and H=4 grid settings before claiming exact reproduction. Likewise, development data paths and update counts differ from the manuscript configuration in places.

See the existing [pipeline notes](../pipeline.md). They document implementation behavior and should be interpreted alongside the specific experiment configuration.

## Reading the reported results

The README table and figures come from the manuscript, not reruns of this public snapshot. Grid training-step plots use accepted actor updates. The reported GPU-hours concern online-training computation. EB-ALFRED uses a frozen world model with behavior-cloning support, so it does not test the same online-refresh loop as the grid ablations.
