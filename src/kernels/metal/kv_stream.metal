// src/kernels/metal/kv_stream.metal - the port of src/kernels/cuda/kv_stream.cu: the residency map of a
// streamed QSA layer (docs/kv-streaming-design.md).  One resolve threadgroup of 1,024 threads walks the
// selections (hits take the epoch + their clock reference bit, misses are claimed by a strong CAS), runs the
// CLOCK sweep for victims, and re-points the page table; the copy kernel then fills the slots from the host
// copy 16 B per thread.
//
// RULE 9 (docs/PORT_METAL/PROGRESS.md): no pointer travels as setBytes bytes.  The CUDA kernels took
// KvStreamMap (seven device pointers + two int64) and Runs (eight pointers + five ints) BY VALUE through the
// argument bytes - on this GPU such pointers do not dereference.  Both structs are FLATTENED: every array is
// its own bound [[buffer(N)]] argument, the counts are scalars, and the Runs loop became one launch per run
// (same bytes copied, the CUDA kernel's per-block loop over runs just unrolled into per-run dispatches).
//
// Atomics: device-space atomics on this toolchain support only relaxed order (the seq_cst spelling does not
// compile), so every device and threadgroup access that the CUDA file made atomic - or raced benignly on
// __shared__ ints - is a relaxed atomic here, with threadgroup_barrier ordering the phases exactly where the
// original had __syncthreads.  atomicCAS is atomic_compare_exchange_WEAK retried until it decides (see the
// claim site) - a spurious failure would leave a block unclaimed and never resident.
#include "strata_port.metalh"
#include <metal_atomic>

constant const int RT = 1024;             // the resolve block (one threadgroup, as on CUDA)
constant const int COPY_THREADS = 128;    // the copy kernel's threadgroup (its grid width rides as a scalar)
constant const int kKvCtlInts = 16;       // kv_stream.hpp

// include/strata/kernels/qsa.hpp's step slots
constant const int kStepWidth = 3;
constant const int kStepCount = 4;

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
// The CUDA original's warp-scan + warp-of-warp-scan, with the threadgroup atomics spelled explicitly.
static inline int block_scan(int v, threadgroup atomic_int* warp_sums, thread int& total, uint tid, uint lane,
                             uint w) {
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = simd_shuffle_up(x, (uint) o);
        if ((int) lane >= o) x += y;
    }
    if (lane == 31) atomic_store_explicit(warp_sums + w, x, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w == 0) {
        int t = atomic_load_explicit(warp_sums + lane, memory_order_relaxed);
        for (int o = 1; o < 32; o <<= 1) {
            const int y = simd_shuffle_up(t, (uint) o);
            if ((int) lane >= o) t += y;
        }
        atomic_store_explicit(warp_sums + lane, t, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = atomic_load_explicit(warp_sums + 31, memory_order_relaxed);
    const int excl = x - v + (w > 0 ? atomic_load_explicit(warp_sums + w - 1, memory_order_relaxed) : 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return excl;
}

// KvStreamMap flattened (rule 9): the seven arrays as bound buffers, the two counts as scalars.
kernel void kv_stream_resolve_kernel(device atomic_int* page_table [[buffer(0)]],
                                     device int* slot_block [[buffer(1)]],
                                     device atomic_int* slot_stamp [[buffer(2)]],
                                     device atomic_int* slot_ref [[buffer(3)]],
                                     device int* ctl [[buffer(4)]],
                                     device int* miss_block [[buffer(5)]],
                                     device int* miss_slot [[buffer(6)]],
                                     constant const int* ids [[buffer(7)]],
                                     constant const int* steps [[buffer(8)]],
                                     constant const int& n_q [[buffer(9)]],
                                     constant const int& cap [[buffer(10)]],
                                     constant const int& page_size [[buffer(11)]],
                                     constant const int& n_slots [[buffer(12)]],
                                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int s_nmiss, s_lookups, s_cut;
    threadgroup atomic_int warp_sums[32];
    const uint lane = tid & 31u, w = tid >> 5;
    const int epoch = ctl[0] + 1;
    if (tid == 0) {
        atomic_store_explicit(&s_nmiss, 0, memory_order_relaxed);
        atomic_store_explicit(&s_lookups, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * kStepCount + kStepWidth];
        constant const int* qi = ids + (ulong) q * (ulong) cap;
        for (int i = (int) tid; i < width; i += RT) {
            const int b = qi[i] / page_size;
            if (i > 0 && qi[i - 1] / page_size == b) continue;   // ids are ascending: one lookup per block
            ++lookups;
            const int sl = atomic_load_explicit(page_table + b, memory_order_relaxed);
            if (sl >= 0) {
                atomic_store_explicit(slot_stamp + sl, epoch, memory_order_relaxed);
                atomic_store_explicit(slot_ref + sl, 1, memory_order_relaxed);
            } else if (sl == -1) {
                // atomicCAS: the toolchain has only the WEAK compare-exchange (relaxed device atomics are all
                // it offers), so a spurious failure is retried until the exchange happens or another thread's
                // claim (-2) shows up - a strong CAS's semantics rebuilt from the weak one.
                // claim (a genuine loss leaves e = -2 and the loop exits without winning)
                bool won = false;
                while (!won) {
                    int e = -1;
                    won = atomic_compare_exchange_weak_explicit(page_table + b, &e, -2, memory_order_relaxed,
                                                                memory_order_relaxed);
                    if (!won && e != -1) break;   // a genuine loss: the block is claimed
                }
                if (won) miss_block[atomic_fetch_add_explicit(&s_nmiss, 1, memory_order_relaxed)] = b;
            }
        }
    }
    atomic_fetch_add_explicit(&s_lookups, lookups, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never
    //    taken; a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = atomic_load_explicit(&s_nmiss, memory_order_relaxed);
    const int n = n_slots;
    int hand = ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int) (((long) hand + (long) tid) % n);
        const bool mine = atomic_load_explicit(slot_stamp + j, memory_order_relaxed) == epoch;
        const bool cand = !mine && (slot_block[j] < 0 ||
                                    atomic_load_explicit(slot_ref + j, memory_order_relaxed) == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total, tid, lane, w);
        const int want = need - got;
        if (tid == 0) atomic_store_explicit(&s_cut, RT, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cand && rank == want - 1)
            atomic_store_explicit(&s_cut, (int) tid + 1, memory_order_relaxed);  // the hand stops just past the
        threadgroup_barrier(mem_flags::mem_threadgroup);                        // last slot taken
        const int cut = atomic_load_explicit(&s_cut, memory_order_relaxed);
        if (cand && rank < want) {
            miss_slot[got + rank] = j;
            atomic_store_explicit(slot_stamp + j, epoch, memory_order_relaxed);  // taken: a sweep that wraps
        } else if (tid < (uint) cut && !mine) {                                 // around must not take it twice
            atomic_store_explicit(slot_ref + j, 0, memory_order_relaxed);
        }
        got += total < want ? total : want;
        hand = (int) (((long) hand + cut) % n);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (int k = (int) tid; k < need; k += RT) {
        const int b = miss_block[k];
        if (k >= placed) {
            atomic_store_explicit(page_table + b, -1, memory_order_relaxed);   // overflow: never happens with a
            continue;                                                          // legal n_slots
        }
        const int sl = miss_slot[k];
        const int old = slot_block[sl];
        if (old >= 0) atomic_store_explicit(page_table + old, -1, memory_order_relaxed);
        slot_block[sl] = b;
        atomic_store_explicit(slot_stamp + sl, epoch, memory_order_relaxed);
        atomic_store_explicit(slot_ref + sl, 1, memory_order_relaxed);
        atomic_store_explicit(page_table + b, sl, memory_order_relaxed);
    }
    if (tid == 0) {
        ctl[0] = epoch;
        ctl[1] = hand;
        ctl[2] = placed;
        if (placed < need) ctl[3] = 1;
        device ulong* c = reinterpret_cast<device ulong*>(ctl + 4);   // the u64 counters at ctl[4..15]
        c[0] += (ulong) placed;
        c[1] += (ulong) atomic_load_explicit(&s_lookups, memory_order_relaxed);
        c[2] += 1ul;
    }
}

// One threadgroup per grid-stride step, one RUN (array) per launch: block k's run copied from the host copy
// into its slot, 16 B per thread - the CUDA kernel's inner `for (a = 0; a < r.n; ++a)` loop, unrolled by the
// launcher so every src/dst is a bound buffer (rule 9).  `len` is a multiple of 16 (kv_stream_resolve
// validates the scale run; the q4/int8/f16 runs are 16-multiples by construction).
kernel void kv_stream_copy_kernel(constant const int* miss_block [[buffer(0)]],
                                  constant const int* miss_slot [[buffer(1)]],
                                  constant const int* ctl [[buffer(2)]],
                                  device const uint8_t* src [[buffer(3)]],      // host-mapped copy: device
                                  device uint8_t* dst [[buffer(4)]],             // space, like the doorbells
                                  constant const int& len [[buffer(5)]],
                                  constant const int& n_groups [[buffer(6)]],
                                  uint g [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    const int need = ctl[2];
    for (int k = (int) g; k < need; k += n_groups) {
        const long b = miss_block[k], sl = miss_slot[k];
        device const uint4* s4 = reinterpret_cast<device const uint4*>(src + (ulong) b * (ulong) len);
        device uint4* d4 = reinterpret_cast<device uint4*>(dst + (ulong) sl * (ulong) len);
        for (int i = (int) tid; i < len / 16; i += COPY_THREADS) d4[i] = s4[i];
    }
}

kernel void kv_stream_reset_kernel(device int* page_table [[buffer(0)]],
                                   device int* slot_block [[buffer(1)]],
                                   device int* slot_stamp [[buffer(2)]],
                                   device int* slot_ref [[buffer(3)]],
                                   device int* ctl [[buffer(4)]],
                                   constant const long& n_blocks [[buffer(5)]],
                                   constant const long& n_slots [[buffer(6)]],
                                   constant const long& total [[buffer(7)]],
                                   uint x [[thread_position_in_grid]]) {
    const long i0 = (long) x, st = total;
    for (long i = i0; i < n_blocks; i += st) page_table[i] = -1;
    for (long i = i0; i < n_slots; i += st) {
        slot_block[i] = -1;
        slot_stamp[i] = -1;
        slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) ctl[i0] = 0;
}

kernel void kv_stream_ring_kernel(device int* table [[buffer(0)]],
                                  constant const long& n_blocks [[buffer(1)]],
                                  constant const long& n_slots [[buffer(2)]],
                                  constant const long& total [[buffer(3)]],
                                  uint x [[thread_position_in_grid]]) {
    for (long i = (long) x; i < n_blocks; i += total) table[i] = (int) (i % n_slots);
}
