# Data

[Back to README](../README.md)

## Release scope

Prepared data is part of the planned initial release. Dataset archives, immutable download URLs, checksums, licenses, and the exact train/validation/calibration/evaluation manifests are not yet included in this repository snapshot. The collection and feature-preparation code is available.

The registered default collection adapters currently cover `sokoban` and `frozenlake`. The five environments evaluated in the manuscript should not be interpreted as five complete public data pipelines in this checkout.

## Collect a small example

Use the installation instructions first. For Sokoban, verify the real environment is available before collection because the adapter also supports a mock development fallback:

```bash
uv run python -c "from sembelief_wm.data.adapters.sokoban import SokobanAdapter; SokobanAdapter(require_real=True).make_env(seed=10000); print('Real Sokoban is available')"
uv run python scripts/train/collect_offline_dataset.py \
  --env-id sokoban \
  --require-real-env \
  --output-dir data/sokoban_example \
  --num-episodes 10 \
  --seed-start 10000 \
  --max-steps 25 \
  --split train \
  --strategy random
```

This collects a small random-policy example, not an exact paper dataset. The CLI accepts repeatable `--strategy NAME[:WEIGHT]` arguments; available strategies depend on the adapter.

## Storage

```text
data/<dataset>/
├── raw/
│   ├── manifest.jsonl
│   └── <episode_id>.pt
└── tokenized/              # Created by feature preparation
    ├── manifest.jsonl
    └── <episode_id>.pt
```

Raw episodes store observations, environment actions, model actions, rewards, termination flags, per-step information, and metadata. A trajectory with T transitions contains T+1 observations. The raw manifest records episode ID, relative path, environment, strategy, length, success, seed, and split.

Paired episodes contain `obs_tokens` from Qwen and separate `semantic_teacher_tokens` from V-JEPA. The teacher targets are not concatenated into the policy’s observation input. For the default 3B configuration, the latent interface has 36 slots of width 2048.

## Paired feature preparation

After setting `MODEL_PATH` to a local Qwen checkpoint, the existing builder accepts:

```bash
uv run python scripts/train/build_qwen_vjepa_teacher_dataset.py \
  --raw-data-dir data/sokoban_example \
  --output-dir data/sokoban_example_paired \
  --backbone-model "$MODEL_PATH" \
  --hidden-dim 2048 \
  --belief-slots 36 \
  --device cuda:0 \
  --dtype bf16 \
  --frame-batch-size 8
```

This step loads both pretrained models and has not been GPU-tested for this documentation draft. The builder preserves the raw manifest’s split labels; it does not create independent evaluation data for you.

## Split discipline and evaluation

Training, validation, reward calibration, and evaluation need separately documented partitions. Choose a new output directory for independent collections: episode IDs are based on environment and seed, and existing IDs are skipped. Do not assume repeated collection with the same seeds changes the split.

The PPO runners expect external evaluation files, including a Sokoban level JSON or a FrozenLake seed JSON. Their paths are configurable through `VAGEN_LEVELS_FILE` and `EVAL_SEEDS_FILE`. These exact files must accompany the release for reproducible benchmark comparisons; fresh random seeds are not a substitute for the reported evaluation protocol.
