# Installation

[Back to README](../README.md)

## Environment

Use Linux with an NVIDIA GPU and a CUDA-compatible PyTorch installation for feature preparation and training. The source declares Python `>=3.10,<3.13` and PyTorch `2.8.0`; `uv.lock` records the resolved dependencies.

```bash
git clone https://github.com/Tom111-lhx/Dwell.git
cd Dwell
uv sync --frozen --python 3.11
uv run python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Install `uv` first if necessary. Commands are intended to run from the repository root. The dependency set includes packages for multiple environments; this is not a minimal grid-only installation.

## Pretrained models

The implementation uses:

- Qwen2.5-VL-3B-Instruct as the VLM backbone.
- `facebook/vjepa2-vitg-fpc64-384-ssv2` as the configured visual teacher.

The shell runners expect `MODEL_PATH` to refer to a **local Qwen directory containing `config.json`**. Set it explicitly to your downloaded model directory; the default is relative to the repository and may not match your model location. The V-JEPA loader uses `VJEPA2Model.from_pretrained` and requires the configured model to be downloadable or cached.

```bash
export MODEL_PATH=/absolute/path/to/Qwen2.5-VL-3B-Instruct
export PYTHON_BIN="$PWD/.venv/bin/python"
export GPU_ID=0
```

Replace the model path before running commands. Model downloads and model-specific terms are separate from this repository’s release.

## Verification status

The README commands and linked entry points were checked against the released source. Installation, model loading, dataset collection, and end-to-end training have **not** been executed in a clean Linux/CUDA environment during this documentation preparation. No minimum GPU memory or training-time guarantee is claimed.

For a small source-interface example, follow [Getting started](../README.md#getting-started). For trained-policy evaluation, first prepare the required checkpoints and exact evaluation files described in the [training guide](reproduction.md).
