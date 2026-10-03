#pragma once
#include <cstddef>

namespace strata::kernels {
// Optional immutable, lossless IQ4_XS decode view. The owning loader must release it
// before freeing the original GGUF allocation, after its graphs have finished.
// STRATA_METAL_IQ4_EXPAND=1 enables preparation; the default is off. No dynamic invalidation:
// weights must remain immutable while the view exists. Multi-token calls use original weights.
// Returns additional allocated bytes, or zero when disabled/over budget/unsupported.
std::size_t metal_iq4_prepare(int type, const void* weights, int n_in, int n_out);
void metal_iq4_release(const void* weights);
}
