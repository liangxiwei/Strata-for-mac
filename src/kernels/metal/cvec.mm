// src/kernels/metal/cvec.mm - the port of src/kernels/cuda/cvec.cu (K13).  The host half is the CUDA
// file's logic with the one-GPU reality of a Mac (its per-device tables collapse); the kernel is
// cvec.metal's.
#include "strata/kernels/cvec.hpp"
#include "strata/platform/metal_launch.hpp"

#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;
bool g_on_host = false;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr; };
DevTables g_dev;
std::vector<float> g_dir_host, g_s_host;
bool upload_here(std::string& err) {
    if (g_dev.dir != nullptr) return true;
    const int flag = g_on_host ? 1 : 0;
    if (cudaMalloc(&g_dev.dir, g_dir_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_dev.s, g_s_host.size() * sizeof(float)) != cudaSuccess || cudaMalloc(&g_dev.on, sizeof(int)) != cudaSuccess ||
        cudaMemcpy(g_dev.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(g_dev.s, g_s_host.data(), g_s_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(g_dev.on, &flag, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "control vector: device allocation failed";
        g_dev = DevTables{};
        return false;
    }
    return true;
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    if (g_dev.dir != nullptr) {           // a new vector replaces the old one
        cudaDeviceSynchronize();
        cudaFree(g_dev.dir);
        cudaFree(g_dev.s);
        cudaFree(g_dev.on);
        g_dev = DevTables{};
    }
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    if (!upload_here(err)) return false;
    g_cvec.dir = g_dev.dir;
    g_cvec.s = g_dev.s;
    g_cvec.on = g_dev.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host) return;
    cudaDeviceSynchronize();             // nothing in flight may still read the flag
    const int v = on ? 1 : 0;
    cudaMemcpy(g_dev.on, &v, sizeof(int), cudaMemcpyHostToDevice);
    g_on_host = on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    if (g_dev.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    metal::Launch k("cvec_kernel", (unsigned) g_cvec.hc, (unsigned) T, 1, THREADS, 1, 1, 0, stream);
    k.buf(R).buf(g_dev.dir).buf(g_dev.s).buf(g_dev.on).scalar(g_cvec.mode).scalar(layer)
     .scalar((int) g_cvec.n_embd).scalar((int) g_cvec.hc).scalar(r_ld)
     .buf(bo).scalar(bo_ld).buf(inj).scalar(inj_ld).scalar(write ? 1 : 0);
    k.done();
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("cvec_apply: launch failed");
}

}  // namespace strata::kernels
