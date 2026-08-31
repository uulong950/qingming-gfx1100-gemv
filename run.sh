#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./build/benchmark}"
OUT="${2:-full_results.csv}"

GEMV_SKIP_PROFILING=1 "$BIN"
mv gemv_results.csv "$OUT"
python3 analyze.py "$OUT" | tee "${OUT%.csv}_summary.txt"
