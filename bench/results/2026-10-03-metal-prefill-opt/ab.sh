#!/bin/bash
# Same-binary prefill A/B on the frozen 10K prompt (no prefix reuse: the first request of a fresh engine), 32 output
# tokens, one engine at a time, two alternating passes. old = this round's prefill kernels off.
set -e
cd "$(dirname "$0")/../../.."
OUT=bench/results/2026-10-03-metal-prefill-opt
RUN="python3 bench/results/2026-10-03-metal-decode-opt/run_decode.py --config /Users/liangxw/src/xproject/petproject/IQ2_XS/strata-mac-optimized.json --tokens 32 --repeats 1"
OFF="--env STRATA_METAL_PREFILL_GEMM2=0 --env STRATA_METAL_PROMPT_ATTN_REG=0 --env STRATA_METAL_PREFILL_MOE2=0"
for pass in a b; do
  if pgrep -f "build-metal/strata --serve" > /dev/null; then echo "another engine is running"; exit 1; fi
  $RUN --output $OUT/old-$pass $OFF > $OUT/old-$pass.log 2>&1
  $RUN --output $OUT/new-$pass > $OUT/new-$pass.log 2>&1
done
echo done
