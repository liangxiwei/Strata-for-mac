// src/platform/metal_smoke.mm - M1 of docs/PORT_METAL/: the three things that must be TRUE before any kernel
// port is trusted, each printed with a measured number on the machine it ran on.
//
//   1. a launch round-trips: a buffer through cudaMalloc -> kernel -> memcpy back is what the kernel wrote;
//   2. the doorbell: a GPU kernel's atomic ring on mapped host memory is visible to a spinning CPU WITHOUT a
//      stream sync (the engine's whole overlap depends on this; RDNA4 needed volatile+fence, see
//      elementwise.cu's ring comment), and a GPU wait kernel observes a CPU-written flag;
//   3. the tape: a captured launch replays with CHANGED buffer contents and picks the change up (replay must
//      read memory, not baked bytes).
//
//   usage: metal_smoke [iters]      (exit 0 = all three hold)
#include "strata/kernels/elementwise.hpp"
#include "strata/platform/metal_launch.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, const char* detail = "") {
    std::printf("  %-42s %s%s%s\n", what, ok ? "ok" : "*** NO ***", *detail ? " (" : "", detail);
    if (! ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 1;
    std::printf("strata metal smoke (M1, docs/PORT_METAL/)\n");

    // ---- 1. a launch round-trips ----------------------------------------------------------
    {
        float* d = nullptr;
        if (cudaMalloc(&d, 16 * sizeof(float)) != cudaSuccess) { check(false, "cudaMalloc"); return 1; }
        float host[16];
        for (int i = 0; i < 16; ++i) host[i] = float(i + 1);
        cudaMemcpy(d, host, sizeof host, cudaMemcpyHostToDevice);
        cudaStream_t s1{};
        cudaStreamCreate(&s1);
        // the SAME launch, spelled directly instead of through elementwise.hpp
        {
            const long nn = 16; const float ss = 2.0f;
            strata::metal::Launch k("scale_kernel", 1, 1, 1, 256, 1, 1, 0, s1);
            k.buf(d).scalar(nn).scalar(ss);
            k.done();
        }
        check(true, "direct Launch spelling ran");
        cudaStreamSynchronize(s1);
        float back_e[16] = {};
        cudaMemcpy(back_e, d, sizeof back_e, cudaMemcpyDeviceToHost);
        int wrong_e = 0;
        for (int i = 0; i < 16; ++i) wrong_e += back_e[i] != float(2 * (i + 1));
        check(wrong_e == 0, "launch round-trip on an explicit stream", "");
        // staging upload (blit) then compute on the SAME buffer - the pattern every engine upload uses,
        // and the one that exposed the sync/visibility rules this runtime now encodes
        float filler[16];
        for (int j = 0; j < 16; ++j) filler[j] = 4.0f;
        cudaMemcpy(d, filler, sizeof filler, cudaMemcpyHostToDevice);   // GPU blit writes d = 4s
        strata::kernels::scale_inplace(d, 16, 3.0f, nullptr);           // -> 12s
        std::printf("    direct CPU read after default-stream scale (want 4..64): %.0f, %.0f\n",
                    ((const float*) d)[0], ((const float*) d)[15]);
        float back[16] = {};
        cudaMemcpy(back, d, sizeof back, cudaMemcpyDeviceToHost);
        int wrong = 0;                                     // 4s (the D2D blit) * 3 (the scale)
        for (int i = 0; i < 16; ++i) wrong += back[i] != 12.0f;
        char detail[64];
        std::snprintf(detail, sizeof detail, "back[0]=%g back[15]=%g want 12", back[0], back[15]);
        check(wrong == 0, "compute-write after blit-write (same buffer)", detail);
        cudaStreamDestroy(s1);
        cudaFree(d);
    }


    // A legacy synchronous read waits for blocking streams but leaves nonblocking work queued.
    {
        float *blocking = nullptr, *independent = nullptr;
        cudaMalloc(&blocking, sizeof(float)); cudaMalloc(&independent, sizeof(float));
        float initial = 3.f;
        cudaMemcpy(blocking, &initial, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(independent, &initial, sizeof(float), cudaMemcpyHostToDevice);
        cudaStream_t a{}, b{};
        cudaStreamCreate(&a); cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
        auto scale = [](float* d, cudaStream_t s) {
            strata::metal::Launch k("scale_kernel", 1, 1, 1, 256, 1, 1, 0, s);
            const long n = 1; const float factor = 2.f;
            k.buf(d).scalar(n).scalar(factor); k.done();
        };
        scale(blocking, a); scale(independent, b);
        float got = 0.f;
        cudaMemcpy(&got, blocking, sizeof(float), cudaMemcpyDeviceToHost);
        check(got == 6.f, "legacy copy observes blocking-stream work");
        check(*independent == 3.f, "legacy copy leaves nonblocking work queued");
        cudaStreamSynchronize(b);
        cudaMemcpy(&got, independent, sizeof(float), cudaMemcpyDeviceToHost);
        check(got == 6.f, "explicit wait observes nonblocking work");
        cudaStreamDestroy(a); cudaStreamDestroy(b);
        cudaFree(blocking); cudaFree(independent);
    }

    // ---- 1b. parity's gdn_gate call shape: four buffers, one scalar, vector uploads ----------
    {
        const int n = 48;
        std::vector<float> alpha(n), dt(n), sa(n), want(n);
        for (int i = 0; i < n; ++i) {
            alpha[i] = -20.0f + float(i) * 2.2f;      // crosses softplus' large-x branch
            dt[i] = float(i) * 0.5f - 8.0f;
            sa[i] = -0.03f - 0.001f * i;
            const double x = double(alpha[i] + dt[i]);
            const double sp = x > 20.0 ? x : std::log1p(std::exp(x));
            want[i] = float(sp * sa[i]);
        }
        float *d_a = nullptr, *d_dt = nullptr, *d_sa = nullptr, *d_g = nullptr;
        cudaMalloc(&d_a, n * 4); cudaMalloc(&d_dt, n * 4); cudaMalloc(&d_sa, n * 4); cudaMalloc(&d_g, n * 4);
        cudaMemcpy(d_a, alpha.data(), n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dt, dt.data(), n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sa, sa.data(), n * 4, cudaMemcpyHostToDevice);
        std::vector<float> seven(n, 7.0f);
        cudaMemcpy(d_g, seven.data(), n * 4, cudaMemcpyHostToDevice);   // 7s: untouched => guard ate it
        strata::kernels::gdn_gate(d_a, d_dt, d_sa, d_g, 1, n, nullptr);
        std::vector<float> got(n, -1.0f);
        cudaMemcpy(got.data(), d_g, n * 4, cudaMemcpyDeviceToHost);
        int wrong = 0, first_bad = -1;
        for (int i = 0; i < n; ++i)
            if (std::abs(got[i] - want[i]) > std::abs(want[i]) * 1e-6 + 1e-30) { ++wrong; if (first_bad < 0) first_bad = i; }
        char detail[96];
        std::snprintf(detail, sizeof detail, "bad=%d@%d got[%d]=%.6g want=%.6g", wrong, first_bad,
                      first_bad < 0 ? 0 : first_bad, first_bad < 0 ? got[0] : got[first_bad],
                      first_bad < 0 ? want[0] : want[first_bad]);
        check(wrong == 0, "gdn_gate (parity's call shape)", detail);
        cudaFree(d_a); cudaFree(d_dt); cudaFree(d_sa); cudaFree(d_g);
    }

    // ---- 2. the doorbell -------------------------------------------------------------------
    {
        uint32_t* seq = nullptr;                                        // mapped host memory
        cudaHostAlloc((void**) &seq, 4096, cudaHostAllocMapped);
        uint32_t* d_seq = nullptr;
        cudaHostGetDevicePointer((uint32_t**) &d_seq, seq, 0);
        *seq = 0;
        // the CPU spins on the ring while the GPU increments it - WITHOUT any sync call
        std::atomic<bool> seen{false};
        std::thread poller([&] {
            while (!seen.load()) {
                if (__atomic_load_n(seq, __ATOMIC_SEQ_CST) == 1) seen.store(true);
            }
        });
        const auto t0 = std::chrono::steady_clock::now();
        strata::kernels::doorbell_ring(d_seq, nullptr);
        // no cudaDeviceSynchronize here on purpose: the poll must see the ring by memory alone
        for (int i = 0; i < 100000 && !seen.load(); ++i) std::this_thread::sleep_for(std::chrono::microseconds(50));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        poller.join();
        char detail[64];
        std::snprintf(detail, sizeof detail, "ring visible to a spinning CPU after %.3f ms (no sync)", ms);
        check(seen.load(), "doorbell ring: GPU -> CPU", detail);

        // the other direction: a GPU wait kernel observes a CPU-written flag, and the work after it runs
        cudaStream_t s{};
        cudaStreamCreate(&s);
        float* d = nullptr;
        cudaMalloc(&d, 8 * sizeof(float));
        uint32_t* flags = nullptr;                                      // flag[0] written by the CPU below
        cudaHostAlloc((void**) &flags, 4096, cudaHostAllocMapped);
        flags[0] = 0;                                                   // flag[1] = the ring value to reach
        uint32_t* d_flags = nullptr;
        cudaHostGetDevicePointer((uint32_t**) &d_flags, flags, 0);
        float zero[8] = {};
        cudaMemcpy(d, zero, sizeof zero, cudaMemcpyHostToDevice);
        strata::kernels::doorbell_wait(d_flags, d_flags + 1, s);        // waits for flag[0] == flag[1]
        flags[1] = 1;                                                   // the target the GPU spins for
        strata::kernels::scale_inplace(d, 8, 3.0f, s);                  // runs after the wait
        flags[0] = 1;                                                   // ring it
        cudaStreamSynchronize(s);
        float back[8] = {};
        cudaMemcpy(back, d, sizeof back, cudaMemcpyDeviceToHost);
        int wrong = 0;
        for (int i = 0; i < 8; ++i) wrong += back[i] != 0.0f;           // zero * 3 == 0: weak; scale to 4 next
        strata::kernels::scale_inplace(d, 8, 4.0f, s);                  // 0 * 4 still 0 - use a fresh buffer
        float* d2 = nullptr;
        cudaMalloc(&d2, 8 * sizeof(float));
        float one[8];
        for (int i = 0; i < 8; ++i) one[i] = 1.0f;
        cudaMemcpy(d2, one, sizeof one, cudaMemcpyHostToDevice);
        strata::kernels::doorbell_wait(d_flags, d_flags + 1, s);
        flags[1] = 2;
        strata::kernels::scale_inplace(d2, 8, 5.0f, s);
        flags[0] = 2;
        cudaStreamSynchronize(s);
        float back2[8] = {};
        cudaMemcpy(back2, d2, sizeof back2, cudaMemcpyDeviceToHost);
        wrong = 0;
        for (int i = 0; i < 8; ++i) wrong += back2[i] != 5.0f;
        check(wrong == 0, "doorbell wait: CPU -> GPU ordering", wrong ? "the work after the wait did not land" : "");
        cudaFree(d); cudaFree(d2); cudaFreeHost(flags); cudaFreeHost(seq); cudaStreamDestroy(s);
    }

    // ---- 3. the tape replays on CHANGED contents --------------------------------------------
    {
        cudaStream_t s{};
        cudaStreamCreate(&s);
        float* d = nullptr;
        cudaMalloc(&d, 8 * sizeof(float));
        cudaGraph_t g{};
        cudaGraphExec_t e{};
        cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        strata::kernels::scale_inplace(d, 8, 2.0f, s);
        cudaStreamEndCapture(s, &g);
        cudaGraphInstantiate(&e, g, nullptr, nullptr, 0);
        cudaGraphDestroy(g);  // MTP destroys the source before the first replay; the exec must own a copy.
        float x[8];
        for (int i = 0; i < 8; ++i) x[i] = float(1 << i);               // 1,2,4..128
        cudaMemcpy(d, x, sizeof x, cudaMemcpyHostToDevice);
        cudaGraphLaunch(e, s);
        cudaStreamSynchronize(s);
        float back[8] = {};
        cudaMemcpy(back, d, sizeof back, cudaMemcpyDeviceToHost);
        int wrong = 0;
        for (int i = 0; i < 8; ++i) wrong += back[i] != 2.0f * float(1 << i);
        check(wrong == 0, "graph replay reads live buffers", wrong ? "replay used stale contents" : "");

        // replay iters times in a row: the ring counts must advance, the doubling must compound
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            cudaGraphLaunch(e, s);
            cudaStreamSynchronize(s);
        }
        const double per_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
        cudaMemcpy(back, d, sizeof back, cudaMemcpyDeviceToHost);
        wrong = 0;
        for (int i = 0; i < 8; ++i) {
            double want = double(1 << i);
            for (int r = 0; r < iters + 1; ++r) want *= 2.0;
            wrong += std::abs(double(back[i]) - want) > std::abs(want) * 1e-6 + 1e-30;
        }
        char detail[80];
        std::snprintf(detail, sizeof detail, "%d replays compound, %.3f ms each (measured)", iters, per_ms);
        check(wrong == 0, "graph replay compounds", detail);
        cudaGraphExecDestroy(e);
        cudaFree(d); cudaStreamDestroy(s);
    }

    // Already-submitted work still belongs to the stream, even with no open command buffer.
    {
        cudaStream_t s{};
        cudaStreamCreate(&s);
        cudaEvent_t event{}, second{};
        cudaEventCreate(&event); cudaEventCreate(&second);
        constexpr int n = 4096;
        float* d = nullptr;
        cudaMalloc(&d, n * sizeof(float));
        for (int repeat = 0; repeat < 8; ++repeat) {
            std::vector<float> input(n, float(repeat + 1));
            cudaMemcpyAsync(d, input.data(), n * sizeof(float), cudaMemcpyHostToDevice, s);
            strata::kernels::scale_inplace(d, n, 3.0f, s);
            cudaEventRecord(event, s);       // commits and clears cur
            cudaEventRecord(second, s);      // must cover the same pending work
            cudaStreamSynchronize(s);
            int wrong = 0;
            for (int i = 0; i < n; ++i) wrong += d[i] != 3.0f * float(repeat + 1);
            check(wrong == 0, "sync waits for previously committed work");
            check(cudaEventQuery(second) == cudaSuccess, "event re-record follows submitted work");
        }
        strata::kernels::scale_inplace(d, n, 2.0f, s);
        cudaEventRecord(event, s);
        cudaEventDestroy(event);             // no completion callback may dereference the freed event
        cudaDeviceSynchronize();
        check(d[n - 1] == 48.0f, "device sync waits after event destruction");
        cudaEventDestroy(second);
        cudaFree(d); cudaStreamDestroy(s);
    }

    std::printf("%s: %d failure%s\n", failures ? "SMOKE FAILED" : "smoke ok", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
