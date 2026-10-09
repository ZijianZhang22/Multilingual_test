#!/usr/bin/env bash
# One-command launcher. Run from anywhere in the repository.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec python -m experiments.literature_measurements.run_all --help
fi
if [[ -z "${ANCHOR:-}" || -z "${ADAPTED:-}" ]]; then
  cat >&2 <<'HELP'
Set ANCHOR and ADAPTED to your local checkpoint directories:
  ANCHOR=/path/to/stage1_en ADAPTED=/path/to/stage2_zh \
    bash experiments/literature_measurements/run_all.sh

For all literature measurements:
  ANCHOR=... ADAPTED=... MODE=full \
    bash experiments/literature_measurements/run_all.sh

Overrides: OUT, CORE_LAYER, LAYERS, MODE, DATA_DIR.
Additional arguments are forwarded to run_all.py.
HELP
  exit 2
fi
args=(
  --anchor "$ANCHOR" --adapted "$ADAPTED"
  --out "${OUT:-mechanism_runs/literature_measurements_3b}"
  --mode "${MODE:-pilot}"
  --core-layer "${CORE_LAYER:-20}"
  --layers "${LAYERS:-6,12,20,24}"
  --data-dir "${DATA_DIR:-invariance_data/wiki}"
)
exec python -m experiments.literature_measurements.run_all "${args[@]}" "$@"
