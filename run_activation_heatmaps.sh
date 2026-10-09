#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/experiments/activation_heatmaps/requirements.txt"
fi
"$PYTHON_BIN" - <<'PY'
try:
    import torch
except ImportError:
    raise SystemExit('Missing PyTorch. Use a RunPod PyTorch image, or install the CUDA-compatible torch build first.')
from packaging.version import Version
if Version(torch.__version__.split('+')[0]) < Version('2.6'):
    raise SystemExit('Use PyTorch >=2.6 for safe loading of the public TinyLlama .bin weights.')
print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
PY
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
"$PYTHON_BIN" "$ROOT_DIR/experiments/activation_heatmaps/run_suite.py" "$@"
