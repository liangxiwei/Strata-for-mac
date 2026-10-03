// src/kernels/metal/sampler.mm - the port of src/kernels/cuda/sampler.cu's host half (K8): the sampler
// chain's three sampled paths (split top_k by default, one block, and engine 0.1.20's kernel behind
// STRATA_OLD_SAMPLER=1) plus the coupled-draft stage/penalize/sample calls.  The host logic is the CUDA
// file's verbatim - the path table, the capture guard, the per-(device, stream) split scratch - and every
// <<<>>> launch is a metal::Launch chain whose .buf/.scalar order equals the kernel's [[buffer(N)]] indices
// exactly (docs/PORT_METAL/PROGRESS.md's unset-binding bug class).
//
// The one launcher-side restructure: the penalty bitmap the greedy/old/one-block kernels sized as DYNAMIC
// SHARED MEMORY lives in a permanent device buffer instead - a ring of cudaMalloc'd slots, handed out in
// turn (sampler.metal's header has the measurements: n_vocab bits are 32 KB at the wide vocabularies, the
// whole of this GPU's threadgroup budget, before the selection arrays even start).  The bitmap is built and
// consumed inside one dispatch, so a captured graph replays a slot's binding without cross-launch state;
// the ring keeps concurrent launches from sharing one.  A grown-out slot is retired, not freed - a captured
// tape may still hold its pointer - exactly like the split scratch below.
#include "strata/kernels/sampler.hpp"
#include "strata/platform/metal_launch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::kernels {
namespace {

// CUDA's int2 (the split candidate unit: (id, float bits)); the shim's cuda_runtime.h has no vector types.
struct CudaI2 {
    int x, y;
};
static_assert(sizeof(CudaI2) == 8, "the split candidate pairs are 8 bytes");
// the mirrored MSL struct rides one setBytes argument; its layout is the C one on both sides
static_assert(sizeof(SamplerParams) == 64, "SamplerParams must ride setBytes as one 64-byte argument");

constexpr int kSelMax = 64;                              // the widest top_k list
constexpr int kSplitBlockSpan = 4 * 32 * 32;             // 4,096 logits per split block
constexpr int kSplitMaxBlocks = 64;                      // lists the merge holds: n_vocab <= 262,144
constexpr int kSplitMaxRows = 64;                        // rows per split launch (the scratch's bound)
constexpr unsigned kSamplerThreads = 1024;               // the block of the greedy/old/one-block kernels

// top_k 1..64 as given; 0 ("off") and anything wider keep 64; never more than the vocabulary
int sampled_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
}

// Which sampled path runs, read once: `STRATA_OLD_SAMPLER=1` is `sampler_kernel` (engine 0.1.20),
// `STRATA_SAMPLER_ONE_BLOCK=1` the one-block kernel; by default the split top_k wherever it applies.
enum class SampledPath { Split, OneBlock, Old };

bool env_flag(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
}

SampledPath sampled_path() {
    static const SampledPath path = env_flag("STRATA_OLD_SAMPLER")         ? SampledPath::Old
                                    : env_flag("STRATA_SAMPLER_ONE_BLOCK") ? SampledPath::OneBlock
                                                                           : SampledPath::Split;
    return path;
}

// A stream being captured into a graph must not reach `split_scratch` (cudaMalloc): it gets the one-block
// kernel, which needs no memory of its own.  The legacy stream cannot be captured.
bool stream_capturing(void* stream) {
    if (stream == nullptr) return false;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing((cudaStream_t) stream, &st) != cudaSuccess) {
        (void) cudaGetLastError();
        return true;
    }
    return st != cudaStreamCaptureStatusNone;
}

// The split's block lists, one buffer per (device, stream): launches on one stream run in order, so a stream
// reuses its buffer with no sync, and two streams never share one.  Grown on demand (at least doubling, up
// to the size of `kSplitMaxRows` rows at the widest vocabulary), never shrunk.  A grown-out buffer is
// retired, not freed: a pointer handed out earlier may still be waiting for its launch, and freeing it
// would need a sync that proves nothing about that thread.  nullptr when the memory cannot be had, and the
// caller falls back to the one-block kernel; a failed grow is remembered.
CudaI2* split_scratch(void* stream, size_t entries) {
    struct Slot {
        int device;
        void* stream;
        CudaI2* ptr;
        size_t entries;
        size_t failed;                       // the smallest size cudaMalloc refused (0: none)
    };
    static std::mutex mu;
    static std::vector<Slot> slots;
    static std::vector<CudaI2*> retired;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mu);
    Slot* slot = nullptr;
    for (Slot& s : slots)
        if (s.device == device && s.stream == stream) slot = &s;
    if (slot == nullptr) {
        slots.push_back({device, stream, nullptr, 0, 0});
        slot = &slots.back();
    }
    if (slot->entries >= entries) return slot->ptr;
    if (slot->failed != 0 && entries >= slot->failed) return nullptr;
    constexpr size_t kCap = (size_t) kSplitMaxRows * (size_t) kSplitMaxBlocks * (size_t) kSelMax;
    size_t want = 2 * slot->entries < kCap ? 2 * slot->entries : kCap;
    if (want < entries) want = entries;
    CudaI2* ptr = nullptr;
    if (cudaMalloc(&ptr, want * sizeof(CudaI2)) != cudaSuccess) {
        (void) cudaGetLastError();
        want = entries;
        if (cudaMalloc(&ptr, want * sizeof(CudaI2)) != cudaSuccess) {
            (void) cudaGetLastError();
            slot->failed = entries;
            return nullptr;
        }
    }
    if (slot->ptr != nullptr) retired.push_back(slot->ptr);
    slot->ptr = ptr;
    slot->entries = want;
    return ptr;
}

// The penalty-bitmap ring (sampler.metal's header says why the bitmap is not threadgroup memory): eight
// grow-only slots cycled per call.  There is no fallback when the memory cannot be had - silently dropping
// the penalties would be silent wrong numbers, so this dies loudly instead.
unsigned int* take_bits(size_t words) {
    static std::mutex mu;
    static unsigned int* ring[8] = {};
    static size_t sz[8] = {};
    static std::vector<unsigned int*> retired;
    static unsigned next = 0;
    std::lock_guard<std::mutex> lock(mu);
    const unsigned i = next++ & 7u;
    if (sz[i] >= words) return ring[i];
    size_t want = 2 * sz[i] < words ? words : 2 * sz[i];
    unsigned int* ptr = nullptr;
    if (cudaMalloc(&ptr, want * sizeof(unsigned int)) != cudaSuccess) {
        (void) cudaGetLastError();
        want = words;
        if (cudaMalloc(&ptr, want * sizeof(unsigned int)) != cudaSuccess) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "sample_tokens: no penalty-bitmap buffer (%zu words)\n", words);
            std::exit(1);
        }
    }
    if (ring[i] != nullptr) retired.push_back(ring[i]);   // retired, not freed: a capture may hold it
    ring[i] = ptr;
    sz[i] = want;
    return ptr;
}

}  // namespace

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n",
                     p.penalty_last_n, (const void*) history, history_len);
        std::exit(1);
    }
    // the penalty bitmap, one n_vocab-bit row per token, in the device ring; null when the launch's own gate
    // leaves every row penalty-free (the kernels' `use_bits` also demands a non-empty window per row)
    unsigned int* bits = nullptr;
    if (history != nullptr && history_len > 0 && p.penalty_last_n > 0)
        bits = take_bits((size_t) n_tokens * (size_t) ((n_vocab + 31) / 32));
    if (p.greedy || p.temperature <= 0.0f) {
        // One block per token, 1,024 threads over the vocabulary.  See `sampler_greedy_kernel`.
        metal::Launch g("sampler_greedy_kernel", (unsigned) n_tokens, 1, 1, kSamplerThreads, 1, 1, 0, stream);
        g.buf(logits)
            .scalar(n_vocab)
            .buf(history)
            .scalar(history_len)
            .scalar(p)
            .scalar(p.penalty_last_n)
            .scalar(kSamplerThreads)
            .buf(bits)
            .buf(out);
        g.done();
    } else if (sampled_path() == SampledPath::Old) {
        // the same block-per-token shape: the selection's k argmax rounds reduce inside the block
        metal::Launch o("sampler_kernel", (unsigned) n_tokens, 1, 1, kSamplerThreads, 1, 1, 0, stream);
        o.buf(logits)
            .scalar(n_vocab)
            .scalar(n_tokens)
            .buf(history)
            .scalar(history_len)
            .scalar(p)
            .scalar(kSamplerThreads)
            .buf(bits)
            .buf(out);
        o.done();
    } else {
        // The split top_k by default: stage 1 over (61 blocks x rows) for 248,320 logits, stage 2 one warp
        // per row.  The one-block kernel when asked for, or when the split cannot run: a wider vocabulary
        // than the merge holds, more than `kSplitMaxRows` rows, a stream under capture, no scratch.
        const int k = sampled_k(p.top_k, n_vocab);
        const int n_blocks = (n_vocab + kSplitBlockSpan - 1) / kSplitBlockSpan;
        CudaI2* scratch = nullptr;
        if (sampled_path() == SampledPath::Split && n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows &&
            !stream_capturing(stream))
            // sized for 16 rows and 64 entries at least, so a verify window or a wider top_k does not regrow it
            scratch = split_scratch(stream, (size_t) (n_tokens > 16 ? n_tokens : 16) * (size_t) n_blocks *
                                                       (size_t) kSelMax);
        if (scratch != nullptr) {
            metal::Launch sp("sampler_split_part_kernel", (unsigned) n_blocks, (unsigned) n_tokens, 1, 128, 1,
                             1, 0, stream);
            sp.buf(logits)
                .scalar(n_vocab)
                .buf(history)
                .scalar(history_len)
                .scalar(p)
                .scalar(k)
                .scalar(n_blocks)
                .buf(scratch);
            sp.done();
            metal::Launch sm("sampler_split_merge_kernel", (unsigned) n_tokens, 1, 1, 32, 1, 1, 0, stream);
            sm.buf(scratch).scalar(n_blocks).scalar(n_vocab).scalar(p).scalar(k).buf(out);
            sm.done();
        } else {
            metal::Launch ob("sampler_one_block_kernel", (unsigned) n_tokens, 1, 1, kSamplerThreads, 1, 1, 0,
                             stream);
            ob.buf(logits)
                .scalar(n_vocab)
                .buf(history)
                .scalar(history_len)
                .scalar(p)
                .scalar(kSamplerThreads)
                .buf(bits)
                .buf(out);
            ob.done();
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "sample_tokens launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

namespace {

int coupled_blocks(int nv) { return (nv + kSplitBlockSpan - 1) / kSplitBlockSpan; }
int coupled_kpart(int nv) { return nv < kSelMax ? nv : kSelMax; }
void coupled_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

size_t coupled_draft_scratch_bytes(int nv) {
    if (nv <= 0 || coupled_blocks(nv) > kSplitMaxBlocks) return 0;
    return (size_t) coupled_blocks(nv) * (size_t) kSelMax * sizeof(CudaI2);
}

void coupled_draft_stage(const SamplerParams* mapped_params, const int32_t* mapped_hist, SamplerParams* params,
                         int32_t* ring, int cap, void* stream) {
    metal::Launch k("coupled_stage_kernel", 1, 1, 1, 256, 1, 1, 0, stream);
    k.buf(mapped_params).buf(mapped_hist).buf(params).buf(ring).scalar(cap);
    k.done();
    coupled_check("coupled_draft_stage");
}

void coupled_draft_sample(float* logits, int nv, const int32_t* sub_to_id, const int32_t* id_to_sub,
                          int id_vocab, const SamplerParams* params, int32_t* ring, int cap, int j,
                          const int32_t* step_rec, void* scratch, int32_t* out_id, float* out_prob,
                          void* stream) {
    const int n_blocks = coupled_blocks(nv), kpart = coupled_kpart(nv);
    if (nv <= 0 || n_blocks > kSplitMaxBlocks || scratch == nullptr) {
        std::fprintf(stderr, "coupled_draft_sample: %d logits need scratch and at most %d blocks\n", nv,
                     kSplitMaxBlocks);
        std::exit(1);
    }
    // the dedup bitmap of the penalize step (nv bits), in the same device ring
    unsigned int* seen = take_bits((size_t) ((nv + 31) / 32));
    metal::Launch pk("coupled_penalize_kernel", 1, 1, 1, 1024, 1, 1, 0, stream);
    pk.buf(logits)
        .scalar(nv)
        .buf(id_to_sub)
        .scalar(id_vocab)
        .buf(params)
        .buf(ring)
        .scalar(cap)
        .scalar(j)
        .buf(seen);
    pk.done();
    coupled_check("coupled_penalize");
    // the selection of the split sampler, unchanged: every 4,096-logit block's first `kpart` (the widest
    // list, since the request's top_k is only known on the device), penalties already applied above
    const SamplerParams none {};
    metal::Launch sp("sampler_split_part_kernel", (unsigned) n_blocks, 1, 1, 128, 1, 1, 0, stream);
    sp.buf(logits)
        .scalar(nv)
        .buf((const int*) nullptr)
        .scalar(0)
        .scalar(none)
        .scalar(kpart)
        .scalar(n_blocks)
        .buf(scratch);
    sp.done();
    coupled_check("coupled_draft split part");
    metal::Launch mg("coupled_merge_kernel", 1, 1, 1, 32, 1, 1, 0, stream);
    mg.buf((const CudaI2*) scratch)
        .scalar(n_blocks)
        .scalar(nv)
        .scalar(kpart)
        .buf(params)
        .buf(step_rec)
        .buf(sub_to_id)
        .buf(ring)
        .scalar(cap)
        .scalar(j)
        .buf(out_id)
        .buf(out_prob);
    mg.done();
    coupled_check("coupled_draft merge");
}

}  // namespace strata::kernels
