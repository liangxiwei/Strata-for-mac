#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
K="$(cd ../../../.. && pwd)/src/kernels/metal"
TMP="$(mktemp -d /tmp/strata-prefill-tail.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
xcrun -sdk macosx metal -w -fno-fast-math -ffp-contract=off -I "$K" rows.metal -o "$TMP/rows.metallib"
clang++ -std=c++20 -O2 -fobjc-arc rows.mm -framework Metal -framework Foundation -o "$TMP/rows"
(cd "$TMP"; ./rows 7) > rows.txt
xcrun -sdk macosx metal -w -fno-fast-math -ffp-contract=off -I "$K" weights.metal -o "$TMP/weights.metallib"
clang++ -std=c++20 -O2 -fobjc-arc weights.mm -framework Metal -framework Foundation -o "$TMP/weights"
(cd "$TMP"; ./weights 7) > weights.txt
xcrun -sdk macosx metal -w -fno-fast-math -ffp-contract=off -I "$K" weights-fine.metal -o "$TMP/weights-fine.metallib"
clang++ -std=c++20 -O2 -fobjc-arc weights-fine.mm -framework Metal -framework Foundation -o "$TMP/weights-fine"
(cd "$TMP"; ./weights-fine 7) > weights-fine.txt
xcrun -sdk macosx metal -w -fno-fast-math -ffp-contract=off -I "$K" dense.metal -o "$TMP/dense.metallib"
clang++ -std=c++20 -O2 -fobjc-arc dense.mm -framework Metal -framework Foundation -o "$TMP/dense"
(cd "$TMP"; ./dense 7) > dense.txt
