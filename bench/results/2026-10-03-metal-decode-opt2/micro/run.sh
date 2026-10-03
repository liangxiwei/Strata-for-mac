#!/bin/bash
# Isolated kernel microbenchmarks (no model, a few hundred MB of GPU memory each). Each harness includes the
# production .metal file verbatim and compares every variant to the original kernel bit for bit before timing.
# Run with no engine loaded; they run one after another. Output: the *.log files next to this script.
set -e
cd "$(dirname "$0")"
K="$(cd ../../../.. && pwd)/src/kernels/metal"
if pgrep -f "build-metal/strata --serve" > /dev/null; then echo "an engine is running; stop it first"; exit 1; fi
build() { xcrun -sdk macosx metal -fno-fast-math -ffp-contract=off -I $K "$1.metal" -o "$2.metallib"; }
build mmvq_variants mmvq && build gr_variants gr && build ex_variants ex && build topk_variants topk
for h in mmvq_bench gr_bench ex_bench topk_bench counter_probe; do
  clang++ -std=c++20 -O2 -fobjc-arc $h.mm -framework Metal -framework Foundation -o /tmp/strata-$h
done
/tmp/strata-counter_probe > counter-probe.log
/tmp/strata-mmvq_bench mmvq.metallib 9 native_iq4_xs_mmvq_kernel native_iq4_xs_expanded native_iq4_xs_direct_r4 vsr44 \
  v105 v116 v118 vtg vdf_tg vpair vdf_r1 vdf_r2 vdf_r8 > mmvq-iq4.log
/tmp/strata-gr_bench 9 > gr.log
/tmp/strata-ex_bench 9 > experts-iq3s.log
/tmp/strata-topk_bench > topk.log
rm -f /tmp/strata-mmvq_bench /tmp/strata-gr_bench /tmp/strata-ex_bench /tmp/strata-topk_bench /tmp/strata-counter_probe *.metallib
echo done
