#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Compare every writable buffer against independent single-token launches, including guards and
// captured graph replays with changed inputs. Token offsets extend well beyond the first 40 KB.
namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, GUARD = 16;
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
struct Workspace {
    Buffer<float> r, ro, bo, ip, lo, rs, io, mix, xn;
    int dr, db, dl, dh;
    Workspace(int nt, bool irregular) :
        r(2 * GUARD + nt * (D + (irregular ? 16 : 0))), ro(r.n),
        bo(2 * GUARD + nt * (N + (irregular ? 16 : 0))),
        ip(2 * GUARD + nt * (HC + (irregular ? 16 : 0))),
        lo(2 * GUARD + nt * (LR + (irregular ? 16 : 0))), rs(ip.n), io(ip.n), mix(bo.n),
        xn(2 * GUARD + nt * D), dr(D + (irregular ? 16 : 0)), db(N + (irregular ? 16 : 0)),
        dl(LR + (irregular ? 16 : 0)), dh(HC + (irregular ? 16 : 0)) {}
};
}
int main(int argc, char** argv) {
    using namespace strata::kernels;
    std::mt19937 rng(7104);
    auto random = [&] { return float(int(rng() % 2001) - 1000) / 1000; };
    std::vector<float> norm(D);
    std::vector<uint16_t> wd(LR * D), wu(D * LR), wi(HC * D);
    for (auto& v : norm) v = .8f + .1f * random();
    for (auto* w : {&wd, &wu, &wi}) for (auto& v : *w) v = bf16_from_f32(.02f * random());
    Buffer<float> dn(D); Buffer<uint16_t> dd(wd.size()), du(wu.size()), di(wi.size());
    dn.put(norm); dd.put(wd); du.put(wu); di.put(wi);
    cudaStream_t stream{}; check(cudaStreamCreate(&stream));
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int failures = 0, cases = 0;
    for (int nt = bench ? 2 : 1; nt <= (bench ? 4 : 8); ++nt) {
      for (int mode = 0; mode < (bench ? 1 : 5); ++mode) {
        // no pending write; in-place; separate output; no inject; irregular layout + mixed flags/eps.
        const bool apply = mode != 0, inject = mode != 3, separate = mode == 2, irregular = mode == 4;
        Workspace ref(nt, irregular), got(nt, irregular);
        auto args = [&](Workspace& w) {
            std::vector<FusedGrArgs> a(nt);
            for (int t = 0; t < nt; ++t) {
                auto& x = a[t];
                x.R = w.r.p + GUARD + t * w.dr;
                x.R_out = (separate ? w.ro.p : w.r.p) + GUARD + t * w.dr;
                x.apply = apply && (!irregular || t % 2 == 0);
                x.bo_prev = x.apply ? w.bo.p + GUARD + t * w.db : nullptr;
                x.inj_prev = x.apply ? w.ip.p + GUARD + t * w.dh : nullptr;
                x.w_norm = dn.p; x.w_down = dd.p; x.w_up = du.p; x.w_inject = inject ? di.p : nullptr;
                x.eps = irregular ? 1e-6f * (t + 1) : 1e-6f;
                x.lo = w.lo.p + GUARD + t * w.dl; x.rs = w.rs.p + GUARD + t * w.dh;
                x.inject_out = inject ? w.io.p + GUARD + t * w.dh : nullptr;
                x.mixed = w.mix.p + GUARD + t * w.db;
            }
            return a;
        };
        auto ar = args(ref), ag = args(got);
        cudaGraph_t graph{}; cudaGraphExec_t exec{};
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        fused_gr_read_multi(ag.data(), nt, got.xn.p + GUARD, stream);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0)); check(cudaGraphDestroy(graph));
        for (int replay = 0; replay < (bench ? 1 : 3); ++replay) {
            std::vector<float> r(ref.r.n, 123.f), bo(ref.bo.n, 123.f), ip(ref.ip.n, 123.f);
            for (int t = 0; t < nt; ++t) {
                for (int j = 0; j < D; ++j) r[GUARD + t * ref.dr + j] = random();
                for (int j = 0; j < N; ++j) bo[GUARD + t * ref.db + j] = random();
                for (int j = 0; j < HC; ++j) ip[GUARD + t * ref.dh + j] = 2 * random();
            }
            for (Workspace* w : {&ref, &got}) {
                w->r.put(r); w->bo.put(bo); w->ip.put(ip);
                for (Buffer<float>* b : {&w->ro, &w->lo, &w->rs, &w->io, &w->mix, &w->xn})
                    b->put(std::vector<float>(b->n, 123.f));
            }
            for (const auto& x : ar) fused_gr_read(x, stream);
            if (replay == 0) fused_gr_read_multi(ag.data(), nt, got.xn.p + GUARD, stream);
            else check(cudaGraphLaunch(exec, stream));
            check(cudaStreamSynchronize(stream));
            Buffer<float>* rb[] = {&ref.r, &ref.ro, &ref.lo, &ref.rs, &ref.io, &ref.mix, &ref.bo, &ref.ip};
            Buffer<float>* gb[] = {&got.r, &got.ro, &got.lo, &got.rs, &got.io, &got.mix, &got.bo, &got.ip};
            for (int b = 0; b < 8; ++b) {
                auto expected = rb[b]->get(), actual = gb[b]->get();
                if (std::memcmp(expected.data(), actual.data(), actual.size() * 4) != 0) {
                    ++failures;
                    std::fprintf(stderr, "Mismatch nt=%d mode=%d replay=%d buffer=%d\n", nt, mode, replay, b);
                }
            }
            const auto scratch = got.xn.get();
            for (int j = 0; j < GUARD; ++j)
                failures += scratch[j] != 123.f || scratch[scratch.size() - 1 - j] != 123.f;
            ++cases;
        }
        if (bench) {
            check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
            const auto start = std::chrono::steady_clock::now();
            int rounds = 0; double seconds;
            do {
                for (int i = 0; i < 100; ++i) check(cudaGraphLaunch(exec, stream));
                check(cudaStreamSynchronize(stream)); rounds += 100;
                seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            } while (seconds < 10);
            const char* batch = std::getenv("STRATA_METAL_HC_BATCH");
            std::printf("HC batch=%s T=%d: %.3f us/read (%d reads, %.2f s)\n",
                        batch ? batch : "default", nt, seconds * 1e6 / rounds, rounds, seconds);
        }
        check(cudaGraphExecDestroy(exec));
      }
    }
    check(cudaStreamDestroy(stream));
    std::printf("Metal fused GR multi: %d cases, %d failures\n", cases, failures);
    return failures != 0;
}
