// src/kernels/metal/s2_expert_grouped.mm - the port of src/kernels/cuda/s2_expert_grouped.cu's host half.
// Same entry points (include/strata/kernels/s2_expert_grouped.hpp), same block geometry, same validation
// and the same old/new kernel choice (the alignment gates and the environment knobs, verbatim - the parity
// test drives both generations through identical inputs and compares them bitwise, and its fallback checks
// 6-8 REQUIRE the new path to decline on the misaligned fixtures).
//
// THE ONE RESTRUCTURE (rule 9, docs/PORT_METAL/PROGRESS.md; iq_kernels.mm's native_expert_grouped is the
// precedent): the grouped kernels' grp_ptr table holds raw device pointers in device memory, and such a
// pointer is not dereferenceable on this GPU.  So `moe_grouped_s2` reads the cap_groups table entries back
// to the host (one small synchronous copy) and launches gu_grouped/down_grouped(_t) once per group with the
// group's blob bound as its own buffer and the group index as a scalar.  The counts (n_groups, grp_start,
// ent_dst, ent_tok) stay device-resident and are read inside the kernels exactly where the CUDA ones read
// them, so per group the arithmetic is the CUDA kernels' own and old vs new stays bitwise.  Because the
// table can be DEVICE-written (`moe_group_resident`'s kernel builds it), the caller's stream is committed
// and waited before the read - a sync the CUDA original does not need, and the reason this entry point is
// not stream-capturable here (nothing captures it: it is the P6 verify-window path; the token graph uses
// the per-hit device-count entries).
//
// `group_resident_kernel` writes the table with the blob base's ADDRESS, which MSL cannot take from a bound
// pointer - the base therefore rides as a scalar integer and the kernel adds the slot offsets to it.  The
// value written is the same one the CUDA kernel computes (the pointer the caller passed), which is what
// Launch::buf() resolves against the runtime's buffer registry on the way back in.
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "strata/platform/metal_launch.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels {
namespace {

// the geometry the launcher needs (the kernel file carries the blob's full restatement)
constexpr int H = 2560;
constexpr int FF = 640;
constexpr int THREADS = 256;
constexpr int GU_ROWS = 32;    // gate/up rows per grouped block: 4 per warp
constexpr int D_ROWS = 64;     // down rows per grouped block: 8 per warp
constexpr int GMAX = 8;        // entries per group
static_assert(GMAX >= kVerifyMaxT, "a verify window's group can exceed GMAX entries");

void check(const char* who, void* stream) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", who, cudaGetErrorString(e));
        std::exit(1);
    }
    // Deliberately NOT synchronising for a non-null stream, exactly as the .cu does not: this is called
    // once per layer from a captured graph's worth of work, and a null-stream sync per launch is the
    // pattern that made a whole round of measurements the driver's cost (Memory/ERRORS.md RC-7).
    (void) stream;
}

// The previous kernels stay selectable for A/B - `STRATA_OLD_GROUPED=1` in the environment, or
// `moe_grouped_select_old` (the parity test runs both generations in one process).  Environment read once,
// on first use; the choice is made at each launch, so a captured graph keeps the kernels it was captured
// with.
std::atomic<int> g_select_old{-1};
std::atomic<int> g_last_path{-1};

bool old_kernels() {
    const int s = g_select_old.load(std::memory_order_relaxed);
    if (s >= 0) return s != 0;
    static const bool env = [] {
        const char* e = std::getenv("STRATA_OLD_GROUPED");
        return e != nullptr && e[0] == '1';
    }();
    return env;
}

// `STRATA_GROUPED_PAIR_MIN_HITS=N`: the per-hit path keeps the previous one-warp-per-row kernels below N
// hits of capacity (bitwise the same either way; default 0).  Ignored while `moe_grouped_select_old`
// forces a choice.
long long pair_min_hits() {
    if (g_select_old.load(std::memory_order_relaxed) >= 0) return 0;
    static const long long n = [] {
        const char* e = std::getenv("STRATA_GROUPED_PAIR_MIN_HITS");
        return e != nullptr ? std::atoll(e) : 0LL;
    }();
    return n;
}

// The new kernels read the activations as aligned words and the per-hit ones also read a blob's codes as
// uint2, so the gates are the .cu's own: 4-byte aligned activations and scratch, 8-byte aligned arena and
// slot size.  Anything else keeps the previous kernels rather than issuing a misaligned load.
bool new_grouped(const void* x_q8_0, const void* scratch) {
    const bool fast = !old_kernels() && ((uintptr_t) x_q8_0 & 3) == 0 && ((uintptr_t) scratch & 3) == 0;
    g_last_path.store(fast ? 1 : 0, std::memory_order_relaxed);
    return fast;
}

bool new_hit(const void* blob_base, long long blob_bytes, const void* x_q8_0, const void* scratch, long long cap) {
    const bool fast = new_grouped(x_q8_0, scratch) && ((uintptr_t) blob_base & 7) == 0 && (blob_bytes & 7) == 0 &&
                      cap >= pair_min_hits();
    g_last_path.store(fast ? 1 : 0, std::memory_order_relaxed);
    return fast;
}

// The per-hit path's two projections, previous or new kernels (one warp per row, or per pair of rows).
void launch_hit_gu(bool fast, const uint8_t* blob_base, const int32_t* slot_index, long long blob_bytes,
                   const uint8_t* x_q8_0, const float* x_scales, float* gate_up, long long cap,
                   const int32_t* d_count, const int32_t* dst_index, int tok_div, void* cs) {
    const int n = (int) cap;
    if (fast) {
        const long long pairs = cap * (long long) FF;
        metal::Launch k("gu_pair_kernel", (unsigned) ((pairs + THREADS / 32 - 1) / (THREADS / 32)), 1, 1, THREADS,
                        1, 1, 0, cs);
        k.buf(blob_base).buf(slot_index).scalar(blob_bytes).buf(x_q8_0).buf(x_scales).buf(gate_up)
         .scalar(n).buf(d_count).buf(dst_index).scalar(tok_div);
        k.done();
    } else {
        const long long rows = cap * 2LL * FF;
        metal::Launch k("gu_kernel", (unsigned) ((rows + THREADS / 32 - 1) / (THREADS / 32)), 1, 1, THREADS, 1, 1,
                        0, cs);
        k.buf(blob_base).buf(slot_index).scalar(blob_bytes).buf(x_q8_0).buf(x_scales).buf(gate_up)
         .scalar(n).buf(d_count).buf(dst_index).scalar(tok_div);
        k.done();
    }
}

void launch_hit_down(bool fast, const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                     long long blob_bytes, const uint8_t* h_q8_0, const float* h_scales, float* out, long long cap,
                     const int32_t* d_count, void* cs) {
    const int n = (int) cap;
    if (fast) {
        const long long pairs = cap * (long long) (H / 2);
        metal::Launch k("down_pair_kernel", (unsigned) ((pairs + THREADS / 32 - 1) / (THREADS / 32)), 1, 1, THREADS,
                        1, 1, 0, cs);
        k.buf(blob_base).buf(slot_index).buf(dst_index).scalar(blob_bytes).buf(h_q8_0).buf(h_scales).buf(out)
         .scalar(n).buf(d_count);
        k.done();
    } else {
        const long long rows = cap * (long long) H;
        metal::Launch k("down_kernel", (unsigned) ((rows + THREADS / 32 - 1) / (THREADS / 32)), 1, 1, THREADS, 1, 1,
                        0, cs);
        k.buf(blob_base).buf(slot_index).buf(dst_index).scalar(blob_bytes).buf(h_q8_0).buf(h_scales).buf(out)
         .scalar(n).buf(d_count);
        k.done();
    }
}

// the shared steps of every grouped call: silu(gate) * up in place, then the intermediate's own Q8_0
// contract (`quantize_q8_0[_scaled]` - the quantize_act port's own kernels, the same functions the .cu calls)
void launch_swiglu_and_quantize(float* gate_up, uint8_t* h_q8_0, float* h_scales, long long pairs,
                                const float* x_scales, void* stream) {
    {
        metal::Launch k("swiglu_kernel", (unsigned) ((pairs + THREADS - 1) / THREADS), 1, 1, THREADS, 1, 1, 0,
                        stream);
        k.buf(gate_up).scalar(pairs);
        k.done();
    }
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, pairs, stream);
    else quantize_q8_0(gate_up, h_q8_0, pairs, stream);
}

// the grouped projections, one launch per group (rule 9; the file comment has the reasoning)
void launch_grouped_gu(bool fast, const uint8_t* blob, const int32_t* grp_start, const int32_t* n_groups,
                       const int32_t* ent_tok, const uint8_t* x_q8_0, const float* x_scales, float* gate_up,
                       int cap_entries, int g, void* cs) {
    metal::Launch k(fast ? "gu_grouped_t_kernel" : "gu_grouped_kernel", (unsigned) (2 * FF / GU_ROWS), 1, 1,
                    THREADS, 1, 1, 0, cs);
    k.buf(blob).buf(grp_start).buf(n_groups).buf(ent_tok).buf(x_q8_0).buf(x_scales).buf(gate_up)
     .scalar(cap_entries).scalar(g);
    k.done();
}

void launch_grouped_down(bool fast, const uint8_t* blob, const int32_t* grp_start, const int32_t* n_groups,
                         const int32_t* ent_dst, const uint8_t* h_q8_0, const float* h_scales, float* out, int g,
                         void* cs) {
    metal::Launch k(fast ? "down_grouped_t_kernel" : "down_grouped_kernel", (unsigned) (H / D_ROWS), 1, 1, THREADS,
                    1, 1, 0, cs);
    k.buf(blob).buf(grp_start).buf(n_groups).buf(ent_dst).buf(h_q8_0).buf(h_scales).buf(out).scalar(g);
    k.done();
}

}  // namespace

void moe_grouped_select_old(int old) { g_select_old.store(old < 0 ? -1 : (old != 0 ? 1 : 0)); }

int moe_grouped_last_path() { return g_last_path.exchange(-1, std::memory_order_relaxed); }

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    // R4.2h: the fp32 scales for the INTERMEDIATE's own quantization, one per 32-element chunk per hit.
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) +
           ((xh + 15) & ~15ull);
}

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    void* cs = stream;
    const bool fast = new_hit(blob_base, blob_bytes, x_q8_0, scratch, n_hits);

    const uint64_t gu_bytes = ((uint64_t) n_hits * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);

    // 1. gate + up, one launch for every row of every hit.
    launch_hit_gu(fast, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, n_hits, nullptr, nullptr, 0,
                  cs);
    check("moe_hit_grouped_s2/gu", stream);
    // 2. silu(gate) * up, 3. the intermediate's own contract (fp32 scales too when the caller supplies
    //    them - otherwise the down projection would keep the disagreement R4.2h removed), 4. down.
    launch_swiglu_and_quantize(gate_up, h_q8_0, h_scales, n_hits * (int64_t) FF, x_scales, stream);
    check("moe_hit_grouped_s2/swiglu+quantize", stream);
    launch_hit_down(fast, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                    x_scales != nullptr ? h_scales : nullptr, out, n_hits, nullptr, cs);
    check("moe_hit_grouped_s2/down", stream);
}

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    if (k < 1 || k > 32) { std::fprintf(stderr, "moe_hit_select: k must be 1..32\n"); std::exit(1); }
    metal::Launch s("hit_select_kernel", 1, 1, 1, 32, 1, 1, 0, stream);
    s.buf(ids).buf(res_row).scalar(k).scalar(n_expert).buf(slot).buf(dst).buf(count);
    s.done();
    check("moe_hit_select", stream);
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    void* cs = stream;
    const bool fast = new_hit(blob_base, blob_bytes, x_q8_0, scratch, cap);
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    launch_hit_gu(fast, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, cap, d_count, nullptr, 0, cs);
    check("moe_hit_grouped_s2_dev/gu", stream);
    launch_swiglu_and_quantize(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, x_scales, stream);
    check("moe_hit_grouped_s2_dev/swiglu+quantize", stream);
    launch_hit_down(fast, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                    x_scales != nullptr ? h_scales : nullptr, out, cap, d_count, cs);
    check("moe_hit_grouped_s2_dev/down", stream);
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot,
                          int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_hit_select_multi: n must be 1..128\n"); std::exit(1); }
    metal::Launch k("hit_select_multi_kernel", 1, 1, 1, 128, 1, 1, 0, stream);
    k.buf(ids).buf(res_row).scalar(n).scalar(n_expert).buf(slot).buf(dst).buf(count);
    k.done();
    check("moe_hit_select_multi", stream);
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    void* cs = stream;
    const bool fast = new_hit(blob_base, blob_bytes, x_q8_0, scratch, cap);
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    launch_hit_gu(fast, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, cap, d_count, dst_index,
                  k_per_token, cs);
    check("moe_hit_grouped_s2_multi/gu", stream);
    launch_swiglu_and_quantize(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, x_scales, stream);
    check("moe_hit_grouped_s2_multi/swiglu+quantize", stream);
    launch_hit_down(fast, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                    x_scales != nullptr ? h_scales : nullptr, out, cap, d_count, cs);
    check("moe_hit_grouped_s2_multi/down", stream);
}

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_group_resident: n must be 1..128\n"); std::exit(1); }
    // the blob base rides as an INTEGER: the kernel writes base + id * blob into the pointer table, and on
    // this backend that table is read back by the host and resolved by Launch::buf() (rule 9) - MSL has no
    // way to take a bound pointer's integer value, so the host passes the number it already has.
    const unsigned long long base_addr = (unsigned long long) (uintptr_t) base;
    metal::Launch k("group_resident_kernel", 1, 1, 1, 128, 1, 1, 0, stream);
    k.buf(ids).scalar(n).scalar(k_per_tok).scalar(base_addr).scalar(blob).buf(grp_ptr).buf(grp_start)
     .buf(counts).buf(ent_dst).buf(ent_tok);
    k.done();
    check("moe_group_resident", stream);
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    void* cs = stream;
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    const bool fast = new_grouped(x_q8_0, scratch);
    // RULE 9: the pointer table is read back to the host so each group's blob can be BOUND (a pointer
    // loaded from device memory is not dereferenceable here).  The table can be device-written
    // (moe_group_resident's kernel), so the caller's stream is committed and waited first; the entries
    // unused by the actual group count read 0 and bind nil, which the kernels' `g >= *n_groups` exit
    // leaves untouched.
    if (stream != nullptr) cudaStreamSynchronize((cudaStream_t) stream);
    std::vector<unsigned long long> ptrs((size_t) cap_groups, 0);
    if (cudaMemcpy(ptrs.data(), grp_ptr, sizeof(unsigned long long) * (size_t) cap_groups, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "moe_grouped_s2: cannot read the group pointer table\n");
        std::exit(1);
    }
    for (int g = 0; g < (int) cap_groups; ++g)
        launch_grouped_gu(fast, (const uint8_t*) ptrs[g], grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up,
                          (int) cap_entries, g, cs);
    check("moe_grouped_s2/gu", stream);
    launch_swiglu_and_quantize(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, x_scales, stream);
    check("moe_grouped_s2/swiglu+quantize", stream);
    const float* hs = x_scales != nullptr ? h_scales : nullptr;
    for (int g = 0; g < (int) cap_groups; ++g)
        launch_grouped_down(fast, (const uint8_t*) ptrs[g], grp_start, n_groups, ent_dst, h_q8_0, hs, out, g, cs);
    check("moe_grouped_s2/down", stream);
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const int n = (int) n_embd;
    const int gx = (int) ((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8);
    metal::Launch k("add_hits_kernel", (unsigned) gx, (unsigned) cap, 1, 256, 1, 1, 0, stream);
    k.buf(parts).buf(hit_out).buf(dst).buf(count).scalar(n).scalar(gx);
    k.done();
    check("moe_hit_add", stream);
}

void moe_hit_grouped_s2_cpu_order(const uint8_t* blob_base, const int32_t* slot_index,
                                 const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes,
                                 const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                                 const float* x_scales, float* gate_up_trace) {
    if (n_hits <= 0) return;
    if (x_scales == nullptr) {
        std::fprintf(stderr, "moe_hit_grouped_s2_cpu_order requires fp32 activation scales\n");
        std::exit(1);
    }
    void* cs = stream;
    const uint64_t gu_bytes = ((uint64_t) n_hits * 2 * FF * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (FF / 32) * 34 + 15) & ~15ull;
    const uint64_t scale_bytes = ((uint64_t) n_hits * (FF / 32) * 4 + 15) & ~15ull;
    float* gu = (float*) scratch;
    uint8_t* hq = (uint8_t*) scratch + gu_bytes;
    float* hs = (float*) (hq + q8_bytes);
    float* hh = (float*) ((uint8_t*) hs + scale_bytes);
    float* xh = (float*) ((uint8_t*) hh + scale_bytes);
    {
        metal::Launch k("activation_correction_kernel", (unsigned) ((H / 32 + THREADS - 1) / THREADS), 1, 1,
                        THREADS, 1, 1, 0, cs);
        k.buf(x_q8_0).buf(x_scales).buf(xh).scalar(H / 32);
        k.done();
    }
    check("cpu_order/input_correction", stream);
    const int rows_per_block = THREADS / 8;
    {
        metal::Launch k("cpu_order_projection_kernel_false",
                        (unsigned) ((n_hits * 2 * FF + rows_per_block - 1) / rows_per_block), 1, 1, THREADS, 1, 1,
                        0, cs);
        k.buf(blob_base).buf(slot_index).buf(dst_index).scalar(blob_bytes).buf(x_q8_0).buf(x_scales).buf(xh)
         .buf(gu).scalar((int) n_hits);
        k.done();
    }
    check("cpu_order/gate_up", stream);
    if (gate_up_trace != nullptr &&
        cudaMemcpyAsync(gate_up_trace, gu, (size_t) n_hits * 2 * FF * sizeof(float),
                        cudaMemcpyDeviceToDevice, (cudaStream_t) cs) != cudaSuccess) {
        std::fprintf(stderr, "cpu_order/gate_up_trace copy failed\n");
        std::exit(1);
    }
    {
        metal::Launch k("cpu_order_swiglu_kernel", (unsigned) ((n_hits * FF + THREADS - 1) / THREADS), 1, 1,
                        THREADS, 1, 1, 0, cs);
        k.buf(gu).scalar((int) (n_hits * FF));
        k.done();
    }
    check("cpu_order/swiglu", stream);
    {
        metal::Launch k("cpu_order_quantize_kernel", (unsigned) ((n_hits * (FF / 32) + THREADS - 1) / THREADS), 1,
                        1, THREADS, 1, 1, 0, cs);
        k.buf(gu).buf(hq).buf(hs).buf(hh).scalar((int) (n_hits * (FF / 32)));
        k.done();
    }
    check("cpu_order/intermediate_quantize", stream);
    {
        metal::Launch k("cpu_order_projection_kernel_true",
                        (unsigned) ((n_hits * H + rows_per_block - 1) / rows_per_block), 1, 1, THREADS, 1, 1, 0,
                        cs);
        k.buf(blob_base).buf(slot_index).buf(dst_index).scalar(blob_bytes).buf(hq).buf(hs).buf(hh).buf(out)
         .scalar((int) n_hits);
        k.done();
    }
    check("cpu_order/down", stream);
}

}  // namespace strata::kernels
