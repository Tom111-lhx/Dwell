# Training scripts

The scripts are organized by ownership rather than by implementation detail.

```text
scripts/
├── frozenlake/   # FrozenLake entry points and task-specific spatial repair
├── sokoban/      # Sokoban entry points
└── train/        # Shared data, WM, reward, critic, and PPO components
```

## Primary entry points

- FrozenLake full pipeline: `scripts/frozenlake/run_frozenlake_full_training.sh`
- Sokoban full pipeline: `scripts/sokoban/run_sokoban_full_training.sh`

Both entry points accept `PHASE=data|base_wm|prior|wm|reward|critic|ppo|rl|all`.
They call the shared components under `scripts/train/`; those components are
kept task-agnostic and should not be duplicated into task directories.

Examples:

```bash
GPU_ID=0 PHASE=all bash scripts/frozenlake/run_frozenlake_full_training.sh
GPU_ID=1 PHASE=all bash scripts/sokoban/run_sokoban_full_training.sh
```

Set `PYTHON_BIN` when the project virtual environment is not located at
`.venv/bin/python`.
