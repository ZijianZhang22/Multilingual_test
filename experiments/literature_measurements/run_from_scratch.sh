#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# Some RunPod images enable legacy hf_transfer without installing the optional
# package. Fall back to Hugging Face's normal download path automatically.
if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]] && ! python -c 'import hf_transfer' >/dev/null 2>&1; then
  echo "[environment] hf_transfer is enabled but unavailable; disabling legacy fast transfer" >&2
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
exec python -m experiments.literature_measurements.from_scratch "$@"
