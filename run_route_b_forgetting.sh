#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/experiments/route_b_forgetting/requirements.txt"
fi
"$PYTHON_BIN" - <<'PY'
import torch
from packaging.version import Version
if Version(torch.__version__.split('+')[0]) < Version('2.6'):
    raise SystemExit('Use a RunPod PyTorch >=2.6 image.')
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
PY
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
"$PYTHON_BIN" "$ROOT_DIR/experiments/route_b_forgetting/route_b.py" run "$@"
