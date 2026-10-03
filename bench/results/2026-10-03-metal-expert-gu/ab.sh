#!/bin/bash
# Same-binary decode A/B for the two-rows-per-warp resident gate/up, one engine at a time, three alternating passes.
set -e
cd "$(dirname "$0")/../../.."
OUT=bench/results/2026-10-03-metal-expert-gu
RUN="python3 bench/results/2026-10-03-metal-decode-opt/run_decode.py --config /Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json --tokens 256 --repeats 4"
for pass in a b c; do
  if pgrep -f "build-metal/strata --serve" > /dev/null; then echo "another engine is running"; exit 1; fi
  $RUN --output $OUT/r1-$pass --env STRATA_METAL_EXPERT_GU_R2=0 > $OUT/r1-$pass.log 2>&1
  $RUN --output $OUT/r2-$pass > $OUT/r2-$pass.log 2>&1
done
echo done
