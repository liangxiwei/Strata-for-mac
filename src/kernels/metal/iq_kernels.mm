// src/kernels/metal/iq_kernels.mm - the port of src/kernels/cuda/iq_kernels.cu's host half.  Same entry
// points (include/strata/kernels/iq_kernels.hpp), same block geometry and validation; the kernel names are
// the CUDA ones with the template argument spelled out (mmvq_kernel_16, mmvq_multi_kernel_16_4, ...).
//
// THE ONE RESTRUCTURE (rule 9, docs/PORT_METAL/PROGRESS.md): the grouped expert kernels' grp_ptr table holds
// raw device pointers in device memory, and a pointer loaded from device memory is NOT a usable device
// pointer on this GPU (it dereferences to zero with no error).  The CUDA kernels read grp_ptr[blockIdx.y];
// here the launcher reads the cap_groups table entries back to the host (one small synchronous copy - the
// values are final, the callers upload the table before any launch of this call), binds each group's blob
// as its OWN buffer argument and launches once per group with the group index as a scalar.  The counts
// (n_groups, grp_start, ent_dst, ent_tok) stay device-resident and the kernels read them exactly where the
// CUDA ones do, so the arithmetic per group is the CUDA kernels' own and old vs new stays bitwise.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// llama.cpp's block structs (layout pinned by the header's static_asserts) - the CUDA file includes the same
// header; only the declarations are needed here (sizes for iq_row_bytes and the scratch math).
#define GGML_COMMON_DECL_CPP
#include "../../../third_party/ggml/ggml-common.h"

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// Launches follow the caller's stream. The runtime's synchronous legacy copies supply blocking-stream
// visibility; nonblocking-stream callers explicitly order their readback, as on CUDA. Synchronizing each
// IQ dequantizer here would serialize every expert's prefill and prevent batching command buffers.

// The formats of each role, one list each so a type cannot be in one switch and missing from another
// (the CUDA file's lists, verbatim).
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(8)
#define STRATA_D_FMTS(X) X(20) X(23) X(42) X(7) X(8)
#define STRATA_MMVQ_FMTS(X) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(7) X(8)

// the types dq_dispatch dequantizes
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
           t == 12 || t == 13 || t == 7 || t == 8;
}

// values per block of the types the launchers take (Fmt<TY>::qk in the CUDA file)
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256; };
template<> struct Fmt<17> { static constexpr int qk = 256; };
template<> struct Fmt<18> { static constexpr int qk = 256; };
template<> struct Fmt<20> { static constexpr int qk = 32; };
template<> struct Fmt<21> { static constexpr int qk = 256; };
template<> struct Fmt<22> { static constexpr int qk = 256; };
template<> struct Fmt<23> { static constexpr int qk = 256; };
template<> struct Fmt<29> { static constexpr int qk = 256; };
template<> struct Fmt<42> { static constexpr int qk = 64; };
template<> struct Fmt<12> { static constexpr int qk = 256; };
template<> struct Fmt<13> { static constexpr int qk = 256; };
template<> struct Fmt<7>  { static constexpr int qk = 32; };
template<> struct Fmt<8>  { static constexpr int qk = 32; };

// Formats with a decode-once Split (iq_kernels.metal's iqk_Split); the others (Q4_K, Q5_K, Q5_1, Q8_0:
// UD-Q4_K_XL) take the per-entry kernels, which call the dot per column exactly as before #242.
template<int TY> inline constexpr bool kSplit = false;
template<> inline constexpr bool kSplit<16> = true;
template<> inline constexpr bool kSplit<17> = true;
template<> inline constexpr bool kSplit<18> = true;
template<> inline constexpr bool kSplit<20> = true;
template<> inline constexpr bool kSplit<21> = true;
template<> inline constexpr bool kSplit<22> = true;
template<> inline constexpr bool kSplit<23> = true;
template<> inline constexpr bool kSplit<29> = true;
template<> inline constexpr bool kSplit<42> = true;

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}
// STRATA_OLD_IQ_MMVQ=1 keeps the per-column kernels (bitwise equal to the new ones; kept for A/B timing)
bool g_old_kernels = env_on("STRATA_OLD_IQ_MMVQ");
// STRATA_METAL_EXPERT_DIRECT=0 keeps the dp4a-word kernels for the Q2_0 resident down projection and the
// IQ3_S single column; the direct kernels (iq_kernels.metal) compute the same bits
bool direct_dots() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_METAL_EXPERT_DIRECT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

// kernel names carry the template argument as a suffix (mmvq_kernel_16); a macro cannot stringify a
// template parameter (it expands before instantiation), so the names are spelled with snprintf
template<int TY>
void launch_mmvq(const uint8_t* W, size_t rb, const void* X, float* y, int n_in, int n_out, int ncols, void* s) {
    unsigned gx = (unsigned) ((n_out + 3) / 4);
    char name[64];
    // the CUDA file's if-constexpr chain as a runtime one: the multi kernels are only instantiated for the
    // kSplit types, and the branches that would name them are never taken for the others
    if (TY == 21 && ncols == 1 && !g_old_kernels && direct_dots()) {
        std::snprintf(name, sizeof name, "mmvq_direct_21_r2");     // two rows per warp
        gx = (unsigned) ((n_out + 7) / 8);
    } else if (!kSplit<TY> || g_old_kernels) {
        std::snprintf(name, sizeof name, "mmvq_kernel_%d", TY);
    } else {
        const int nc = ncols <= 1 ? 1 : ncols == 2 ? 2 : ncols <= 4 ? 4 : 8;   // 8 at a time past 8
        std::snprintf(name, sizeof name, "mmvq_multi_kernel_%d_%d", TY, nc);
    }
    metal::Launch k(name, gx, 1, 1, 32, 4, 1, 0, s);
    k.buf(W).scalar(rb).buf(X).buf(y).scalar(n_in).scalar(n_out).scalar(ncols);
    k.done();
}

constexpr int GU_ROWS = 8;     // rows per block (one warp each)

template<int TG>
void launch_gu(unsigned gx, void* s, const uint8_t* blob, const int32_t* grp_start, const int32_t* n_groups,
               const int32_t* ent_tok, const block_q8_1* X, const NativeExpertLayout& L, float* gate, float* up,
               int g) {
    char name[64];
    std::snprintf(name, sizeof name, "native_gu_%skernel_%d", (kSplit<TG> && !g_old_kernels) ? "multi_" : "", TG);
    metal::Launch k(name, gx, 1, 1, 256, 1, 1, 0, s);
    k.buf(blob).buf(grp_start).buf(n_groups).buf(ent_tok).buf(X)
     .scalar(L.n_embd).scalar(L.n_ff).scalar(L.gu_row).scalar(L.up_off).buf(gate).buf(up).scalar(g);
    k.done();
}

template<int TD>
void launch_down(unsigned gx, void* s, const uint8_t* blob, const int32_t* grp_start, const int32_t* n_groups,
                 const int32_t* ent_dst, const block_q8_1* hq, const NativeExpertLayout& L, float* out, int g) {
    char name[64];
    std::snprintf(name, sizeof name, "native_down_%skernel_%d", (kSplit<TD> && !g_old_kernels) ? "multi_" : "", TD);
    metal::Launch k(name, gx, 1, 1, 256, 1, 1, 0, s);
    k.buf(blob).buf(grp_start).buf(n_groups).buf(ent_dst).buf(hq)
     .scalar(L.n_embd).scalar(L.n_ff).scalar(L.d_row).scalar(L.down_off).buf(out).scalar(g);
    k.done();
}

int gu_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_GU_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_D_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}

}  // namespace

void iq_set_old_kernels(bool old) { g_old_kernels = old; }
bool iq_old_kernels() { return g_old_kernels; }

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30; }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 30: return (size_t) n * 2;   // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    metal::Launch k("quantize_q8_1_kernel", (unsigned) ((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    k.buf(x).buf(y).scalar(n);
    k.done();
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    void* s = stream;
    const auto* W = (const uint8_t*) w;
    switch (t) {
#define STRATA_MMVQ(T) case T: launch_mmvq<T>(W, rb, x_q8_1, y, n_in, n_out, ncols, s); break;
        STRATA_MMVQ_FMTS(STRATA_MMVQ)
#undef STRATA_MMVQ
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    metal::Launch k("dequant_flat_kernel_f16", (unsigned) (n / 256), 1, 1, 32, 1, 1, 0, stream);
    k.scalar(t).buf(src).buf(dst);
    k.done();
    check("iq_dequant_f16");
}

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    metal::Launch k("embed_rows_kernel", (unsigned) (n_embd / 256), (unsigned) n_tok, 1, 32, 1, 1, 0, stream);
    k.scalar(t).buf(table).scalar(row_bytes).buf(tokens).scalar(n_embd).buf(out);
    k.done();
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    metal::Launch k("dequant_flat_kernel_f32", (unsigned) (n / 256), 1, 1, 32, 1, 1, 0, stream);
    k.scalar(t).buf(src).buf(dst);
    k.done();
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    // checked like the other entry points: an unknown type used to leave `dst` unwritten, a wrong prompt and no error
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_gu_f16: type %d / %lld\n", t, (long long) n_embd); std::exit(1); }
    const int64_t per_row = n_embd / 256;
    metal::Launch k("dequant_gu_kernel", (unsigned) (n_ff * per_row), 2, 1, 32, 1, 1, 0, stream);
    k.scalar(t).buf(gate).buf(up).scalar(per_row).buf(dst);
    k.done();
    check("iq_dequant_gu_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    void* s = stream;
    // RULE 9: grp_ptr holds raw device pointers in device memory - not dereferenceable here.  Read the
    // table back (one small synchronous copy; the callers upload it before this call), then bind each
    // group's blob as its own buffer argument and launch once per group.  Unused grid rows (g >=
    // *n_groups) keep the CUDA kernel's early exit: the blob is nil there and the kernel returns before
    // touching it.
    std::vector<unsigned long long> ptrs((size_t) cap_groups, 0);
    if (cudaMemcpy(ptrs.data(), grp_ptr, sizeof(unsigned long long) * (size_t) cap_groups, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "native_expert_grouped: cannot read the group pointer table\n");
        std::exit(1);
    }
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const unsigned ggu = (unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS);
    for (int g = 0; g < (int) cap_groups; ++g) {
        const uint8_t* blob = (const uint8_t*) ptrs[g];
        switch (L.gu_type) {
#define STRATA_GU(T) case T: launch_gu<T>(ggu, s, blob, grp_start, n_groups, ent_tok, X, L, gate, up, g); break;
        STRATA_GU_FMTS(STRATA_GU)
#undef STRATA_GU
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
        }
    }
    check("native_expert_grouped/gu");
    const long long nh = (long long) cap_entries * L.n_ff;
    {
        metal::Launch k("swiglu_entries_kernel", (unsigned) ((nh + 255) / 256), 1, 1, 256, 1, 1, 0, s);
        k.buf(gate).buf(up).buf(h).scalar(nh);
        k.done();
    }
    {
        metal::Launch k("quantize_q8_1_kernel", (unsigned) ((nh + 255) / 256), 1, 1, 256, 1, 1, 0, s);
        k.buf(h).buf(hq).scalar(nh);
        k.done();
    }
    const unsigned gd = (unsigned) ((L.n_embd + 7) / 8);
    for (int g = 0; g < (int) cap_groups; ++g) {
        const uint8_t* blob = (const uint8_t*) ptrs[g];
        switch (L.d_type) {
#define STRATA_DOWN(T) case T: launch_down<T>(gd, s, blob, grp_start, n_groups, ent_dst, hq, L, out, g); break;
        STRATA_D_FMTS(STRATA_DOWN)
#undef STRATA_DOWN
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
        }
    }
    check("native_expert_grouped/down");
}

bool native_expert_gemm_supported(const NativeExpertLayout& L) {
    const bool gu = L.gu_type == 16 || L.gu_type == 17 || L.gu_type == 18 || L.gu_type == 21 ||
                    L.gu_type == 22 || L.gu_type == 23 || L.gu_type == 29 || L.gu_type == 42;
    const bool down = L.d_type == 20 || L.d_type == 23 || L.d_type == 42;
    return gu && down && L.n_embd > 0 && L.n_embd % 256 == 0 && L.n_ff > 0 && L.n_ff % 32 == 0 &&
           (L.d_type != 23 || L.n_ff % 256 == 0) && (L.d_type != 42 || L.n_ff % 64 == 0);
}

int native_expert_gemm_rows(double rows_per_expert) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_METAL_PREFILL_MOE2");   // =0: the 16 x 32 tiles
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (!on) return 0;
    return rows_per_expert >= 40.0 ? 32 : 16;
}

void native_expert_gemm(const NativeExpertLayout& L, const uint16_t* x, const uint8_t* arena,
                        const int32_t* tiles, float* y, int64_t n_tiles, bool down, void* stream, int tile_rows) {
    if (n_tiles <= 0) return;
    if (!native_expert_gemm_supported(L)) throw std::invalid_argument("native expert GEMM format/shape");
    char name[64];
    const int64_t cols = down ? L.n_embd : 2 * L.n_ff;
    if (tile_rows == 16 || tile_rows == 32) {   // descriptors of up to tile_rows rows (prefill.cpp builds them)
        std::snprintf(name, sizeof name, "native_gemm2_%s_%d_r%d", down ? "down" : "gu", down ? L.d_type : L.gu_type,
                      tile_rows);
        metal::Launch q(name, (unsigned) ((cols + 63) / 64), (unsigned) n_tiles, 1, tile_rows == 32 ? 256 : 128, 1, 1, 0,
                        stream);
        q.buf(x).buf(arena).buf(tiles).buf(y).scalar(L.n_embd).scalar(L.n_ff)
         .scalar(down ? L.d_row : L.gu_row).scalar(down ? L.down_off : L.up_off);
        q.done();
        check("native_expert_gemm");
        return;
    }
    std::snprintf(name, sizeof name, "native_gemm_%s_%d", down ? "down" : "gu", down ? L.d_type : L.gu_type);
    metal::Launch q(name, (unsigned) ((cols + 31) / 32), (unsigned) n_tiles, 1, 128, 1, 1, 0, stream);
    q.buf(x).buf(arena).buf(tiles).buf(y).scalar(L.n_embd).scalar(L.n_ff)
     .scalar(down ? L.d_row : L.gu_row).scalar(down ? L.down_off : L.up_off);
    q.done();
    check("native_expert_gemm");
}

void native_expert_resident(const NativeExpertLayout& L, const uint8_t* arena, const uint64_t* slot_offsets,
                            int64_t slot_bytes, const int32_t* ids, const int32_t* residency, int n_expert,
                            int n_tok, int k, const void* x_q8_1, void* scratch, float* out, void* stream) {
    if (n_tok <= 0 || k <= 0) return;
    const int cap = n_tok * k, offsets = slot_offsets != nullptr;
    const size_t f = (size_t) cap * L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    void* hq = (uint8_t*) scratch + 3 * fa;
    char name[64];
    std::snprintf(name, sizeof name, "native_resident_gu_%d", L.gu_type);
    metal::Launch gu(name, (unsigned) ((2 * L.n_ff + 7) / 8), (unsigned) cap, 1, 256, 1, 1, 0, stream);
    gu.buf(arena).buf(slot_offsets).buf(ids).buf(residency).buf(x_q8_1)
      .scalar(L.n_embd).scalar(L.n_ff).scalar(L.gu_row).scalar(L.up_off).scalar((uint64_t) slot_bytes)
      .scalar(n_expert).scalar(k).scalar(offsets).buf(gate).buf(up);
    gu.done();
    const long long nh = (long long) cap * L.n_ff;
    metal::Launch sw("swiglu_entries_kernel", (unsigned) ((nh + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    sw.buf(gate).buf(up).buf(h).scalar(nh);
    sw.done();
    metal::Launch q("quantize_q8_1_kernel", (unsigned) ((nh + 255) / 256), 1, 1, 256, 1, 1, 0, stream);
    q.buf(h).buf(hq).scalar(nh);
    q.done();
    const bool direct = L.d_type == 42 && direct_dots();         // four rows per warp
    static const bool specialize_down = [] {
        const char* e = std::getenv("STRATA_METAL_DECODE_DOWN_DIMS"); // =0: runtime-dimension kernel
        return e == nullptr || std::atoi(e) != 0;
    }();
    const bool canonical_down = specialize_down && L.n_embd == 2560 && L.n_ff == 640;
    if (direct) std::snprintf(name, sizeof name, canonical_down ? "native_resident_down_dim_42_r4"
                                                               : "native_resident_down_direct_42_r4");
    else std::snprintf(name, sizeof name, "native_resident_down_%d", L.d_type);
    metal::Launch down(name, (unsigned) ((L.n_embd + (direct ? 31 : 7)) / (direct ? 32 : 8)), (unsigned) cap, 1, 256, 1, 1, 0,
                       stream);
    down.buf(arena).buf(slot_offsets).buf(ids).buf(residency).buf(hq)
        .scalar(L.n_embd).scalar(L.n_ff).scalar(L.d_row).scalar(L.down_off).scalar((uint64_t) slot_bytes)
        .scalar(n_expert).scalar(k).scalar(offsets).buf(out);
    down.done();
    check("native_expert_resident");
}

}  // namespace strata::kernels
