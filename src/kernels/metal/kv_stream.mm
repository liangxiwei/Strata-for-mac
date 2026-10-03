// src/kernels/metal/kv_stream.mm - the port of src/kernels/cuda/kv_stream.cu's host half.  Every kernel
// argument the CUDA file passed inside KvStreamMap/Runs bytes is a BOUND buffer or a scalar here (rule 9,
// docs/PORT_METAL/PROGRESS.md): the resolve kernel takes the seven arrays as buffers plus four int scalars,
// the reset/ring kernels five buffers plus three long scalars, and the copy kernel - one RUN of the CUDA
// original's per-block loop, launched once per array - takes its own src/dst pair.  kv_ring_restore,
// kv_stage_from_host and kv_stream_counters stay cudaMemcpy calls exactly as on CUDA (the shim's copies are
// stream-ordered blits over the same registered buffers).
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/platform/metal_launch.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_stream: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int RT = 1024;   // the resolve block
constexpr int RESET_GROUPS = 128, RESET_THREADS = 256;
constexpr int COPY_GROUPS = 96, COPY_THREADS = 128;
constexpr int RING_GROUPS = 64, RING_THREADS = 256;

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    metal::Launch k("kv_stream_reset_kernel", RESET_GROUPS, 1, 1, RESET_THREADS, 1, 1, 0, stream);
    k.buf(m.page_table).buf(m.slot_block).buf(m.slot_stamp).buf(m.slot_ref).buf(m.ctl)
     .scalar((long) m.n_blocks)
     .scalar((long) m.n_slots)
     .scalar((long) RESET_GROUPS * RESET_THREADS);
    k.done();
    check("reset");
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    // one sweep step looks at RT consecutive slots `(hand + thread) % n_slots`; with fewer slots than RT two
    // threads see the same slot and may both take it for two different misses.  The engine never streams with fewer
    // than qsa_kv_resident_min() / page_size = 5,120 slots, so this is a guard, not a limit.
    if (m.n_slots < RT) {
        std::fprintf(stderr, "kv_stream: %lld slots is fewer than the resolve block (%d): the clock sweep would take a "
                             "slot twice\n",
                     (long long) m.n_slots, RT);
        std::exit(1);
    }
    // KvStreamMap flattened: the seven arrays as bound buffers (rule 9), the counts as scalars
    metal::Launch r("kv_stream_resolve_kernel", 1, 1, 1, RT, 1, 1, 0, stream);
    r.buf(m.page_table).buf(m.slot_block).buf(m.slot_stamp).buf(m.slot_ref).buf(m.ctl)
     .buf(m.miss_block).buf(m.miss_slot).buf(ids).buf(steps)
     .scalar((int) n_q)
     .scalar((int) cap)
     .scalar((int) s.page_size)
     .scalar((int) m.n_slots);
    r.done();
    check("resolve");
    // the CUDA copy kernel loops the runs inside one launch; here each run is its own launch with its own
    // bound src/dst pair - same bytes, and no pointer ever rides the argument bytes
    const Runs rn = runs_of(slots, host, fmt, s);
    for (int a = 0; a < rn.n; ++a) {
        metal::Launch c("kv_stream_copy_kernel", COPY_GROUPS, 1, 1, COPY_THREADS, 1, 1, 0, stream);
        c.buf(m.miss_block).buf(m.miss_slot).buf(m.ctl).buf(rn.src[a]).buf(rn.dst[a])
         .scalar(rn.len[a])
         .scalar(COPY_GROUPS);
        c.done();
    }
    check("copy");
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    metal::Launch k("kv_stream_ring_kernel", RING_GROUPS, 1, 1, RING_THREADS, 1, 1, 0, stream);
    k.buf(page_table)
     .scalar((long) n_blocks)
     .scalar((long) n_slots)
     .scalar((long) RING_GROUPS * RING_THREADS);
    k.done();
    check("ring table");
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
                check("ring restore");
        b += run;
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]), cudaMemcpyDefault,
                            (cudaStream_t) stream) != cudaSuccess)
            check("stage");
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return r;
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace strata::kernels
