#!/usr/bin/env bash
# Run after the private engine exits; each variant checks bitwise outputs before timing.
set -euo pipefail
cd "$(dirname "$0")"
K="$(cd ../../../.. && pwd)/src/kernels/metal"
TMP="$(mktemp -d /tmp/strata-prefill-micro.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
for shader in tiles cache; do
  xcrun -sdk macosx metal -w -fno-fast-math -ffp-contract=off -I "$K" "$shader.metal" -o "$TMP/$shader.metallib"
done
for harness in tiles cache cache64 adaptive; do
  clang++ -std=c++20 -O2 -fobjc-arc "$harness.mm" -framework Metal -framework Foundation -o "$TMP/$harness"
  (cd "$TMP"; "./$harness" 5) > "$harness.txt"
done
