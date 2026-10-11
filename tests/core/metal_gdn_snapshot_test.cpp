#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int S = 128, PAD = 16;
void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
template<class T> struct Buffer {
    T* p{}; size_t n;
    explicit Buffer(size_t size) : n(size) { check(cudaMalloc((void**)&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    void put(const std::vector<T>& v) { check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const {
        std::vector<T> v(n); check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
};
}
int main(int argc, char** argv) {
    using namespace strata::kernels;
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const bool use_snapshot = argc > 2 && std::strcmp(argv[2], "snapshot") == 0;
    std::mt19937 rng(4281);
    auto random = [&] { return float(int(rng() % 2001) - 1000) / 1000; };
    cudaStream_t stream{}; check(cudaStreamCreate(&stream));
    int failures = 0, cases = 0;
    // The existing replay kernel's permanent scratch ring is sized by its first use.
    for (int hv : (bench ? std::vector<int>{48} : std::vector<int>{48, 4})) {
      const int hk = hv == 4 ? 2 : 16, C = S * (2 * hk + hv), V = S * hv, NF = S * hv * S;
      for (int nt = bench ? 2 : 1; nt <= (bench ? 4 : (hv == 4 ? 8 : 4)); ++nt) {
        Buffer<float> ref(NF + 2 * PAD), got(ref.n), snapshots(nt * NF + 2 * PAD);
        Buffer<float> h(nt * C), gate(nt * hv), beta(nt * hv), z(nt * V), gamma(S), yr(nt * V), yg(yr.n);
        Buffer<int32_t> keep(1);
        auto verify = [&](bool snapshot) {
            if (snapshot) gdn_step_norm_snapshots(got.p + PAD, h.p, C, gate.p, beta.p, z.p, gamma.p,
                                                 1e-6f, yg.p, hk, hv, nt, snapshots.p + PAD, stream);
            else gdn_step_norm_multi(got.p + PAD, h.p, C, gate.p, beta.p, z.p, gamma.p,
                                     1e-6f, yg.p, hk, hv, nt, nullptr, stream);
        };
        auto commit = [&](bool snapshot) {
            if (snapshot) gdn_commit_snapshot(got.p + PAD, snapshots.p + PAD, hv, keep.p, stream);
            else gdn_step_norm_multi(got.p + PAD, h.p, C, gate.p, beta.p, z.p, gamma.p,
                                     1e-6f, yg.p, hk, hv, nt, keep.p, stream);
        };
        cudaGraph_t graph{}; cudaGraphExec_t ve{}, ce{};
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); verify(true);
        check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&ve, graph, nullptr, nullptr, 0));
        check(cudaGraphDestroy(graph));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); commit(true);
        check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&ce, graph, nullptr, nullptr, 0));
        check(cudaGraphDestroy(graph));
        for (int replay = 0; replay < (bench ? 1 : 3); ++replay) {
            std::vector<float> state(ref.n, 123.f), hh(h.n), gg(gate.n), bb(beta.n), zz(z.n), gn(S);
            for (int i = PAD; i < PAD + NF; ++i) state[i] = .02f * random();
            for (auto& v : hh) v = .05f * random();
            for (auto& v : gg) v = -.3f + .1f * random();
            for (auto& v : bb) v = .5f + .1f * random();
            for (auto& v : zz) v = random();
            for (auto& v : gn) v = .8f + .1f * random();
            ref.put(state); got.put(state); h.put(hh); gate.put(gg); beta.put(bb); z.put(zz); gamma.put(gn);
            snapshots.put(std::vector<float>(snapshots.n, 123.f));
            gdn_step_norm_multi(ref.p + PAD, h.p, C, gate.p, beta.p, z.p, gamma.p,
                                 1e-6f, yr.p, hk, hv, nt, nullptr, stream);
            if (replay == 0) verify(true); else check(cudaGraphLaunch(ve, stream));
            check(cudaStreamSynchronize(stream));
            auto expected_y = yr.get(), actual_y = yg.get(), actual_state = got.get(), snap = snapshots.get();
            failures += std::memcmp(expected_y.data(), actual_y.data(), yr.n * 4) != 0;
            failures += std::memcmp(state.data(), actual_state.data(), state.size() * 4) != 0;
            for (int i = 0; i < PAD; ++i) failures += snap[i] != 123.f || snap[snap.size() - 1 - i] != 123.f;
            for (int accepted = 0; accepted <= nt; ++accepted) {
                ref.put(state); got.put(state); keep.put({accepted});
                gdn_step_norm_multi(ref.p + PAD, h.p, C, gate.p, beta.p, z.p, gamma.p,
                                     1e-6f, yr.p, hk, hv, nt, keep.p, stream);
                check(cudaGraphLaunch(ce, stream)); check(cudaStreamSynchronize(stream));
                auto expected = ref.get(), actual = got.get();
                if (std::memcmp(expected.data(), actual.data(), ref.n * 4) != 0) {
                    ++failures; std::fprintf(stderr, "GDN state mismatch hv=%d T=%d keep=%d replay=%d\n",
                                             hv, nt, accepted, replay);
                }
                if (accepted > 0) failures += std::memcmp(expected.data() + PAD,
                                                         snap.data() + PAD + (accepted - 1) * NF, NF * 4) != 0;
                ++cases;
            }
        }
        if (bench) {
            keep.put({nt - 1});
            cudaGraphExec_t be{};
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            verify(use_snapshot); commit(use_snapshot);
            check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&be, graph, nullptr, nullptr, 0));
            check(cudaGraphDestroy(graph)); check(cudaGraphLaunch(be, stream)); check(cudaStreamSynchronize(stream));
            const auto start = std::chrono::steady_clock::now(); int rounds = 0; double seconds;
            do {
                for (int i = 0; i < 20; ++i) check(cudaGraphLaunch(be, stream));
                check(cudaStreamSynchronize(stream)); rounds += 20;
                seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            } while (seconds < 10);
            std::printf("GDN %s T=%d keep=%d: %.3f us/verify+commit (%d reads, %.2f s)\n",
                        use_snapshot ? "snapshot" : "replay", nt, nt - 1, seconds * 1e6 / rounds, rounds, seconds);
            check(cudaGraphExecDestroy(be));
        }
        check(cudaGraphExecDestroy(ve)); check(cudaGraphExecDestroy(ce));
      }
    }
    check(cudaStreamDestroy(stream));
    std::printf("Metal GDN snapshots: %d accepted-prefix cases, %d failures\n", cases, failures);
    return failures != 0;
}
