#!/bin/bash
# Isolated prefill kernel microbenchmarks (no model). Each harness includes the production .metal verbatim and
# compares every variant to the original kernel bit for bit before timing. Run with no engine loaded.
set -e
cd "$(dirname "$0")"
K="$(cd ../../../.. && pwd)/src/kernels/metal"
if pgrep -f "build-metal/strata --serve" > /dev/null; then echo "an engine is running; stop it first"; exit 1; fi
build() { xcrun -sdk macosx metal -w -fno-fast-math -ffp-contract=off -I "$K" "$1.metal" -o "$2.metallib"; }
build gemm_variants gemm && build attn_variants attn && build moe_variants moe
for h in gemm_bench attn_bench moe_bench; do
  clang++ -std=c++20 -O2 -fobjc-arc $h.mm -framework Metal -framework Foundation -o /tmp/strata-$h
done
/tmp/strata-gemm_bench 5 > prefill-gemm.log
/tmp/strata-attn_bench 5 > prompt-attn.log
/tmp/strata-moe_bench 5 > prefill-moe.log
rm -f /tmp/strata-gemm_bench /tmp/strata-attn_bench /tmp/strata-moe_bench *.metallib
echo done
