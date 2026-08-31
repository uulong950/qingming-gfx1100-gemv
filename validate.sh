#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
PERF_CSV="${PERF_CSV:-full_results.csv}"
NUM_DIR="${NUM_DIR:-numerical_results}"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100

cmake --build "$BUILD_DIR" -j

./run.sh "./$BUILD_DIR/benchmark" "$PERF_CSV"

rm -rf "$NUM_DIR"
"./$BUILD_DIR/fp32_accuracy" --out-dir "$NUM_DIR"

echo
echo "Performance:"
echo "  $PERF_CSV"
echo "  ${PERF_CSV%.csv}_summary.txt"
echo "Numerical:"
echo "  $NUM_DIR/numerical_report.md"
