#include "gemv.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::int64_t M = 64;
constexpr std::uint64_t SEED = 12345;
constexpr double ATOL = 1.0e-4;
constexpr double RTOL = 1.0e-4;
constexpr double REL_EPS = 1.0e-30;
constexpr double U32 = 5.9604644775390625e-8;
constexpr std::uint64_t LT_MAX_WORKSPACE = 64ULL << 20;
constexpr int LT_HEURISTIC_COUNT = 4;
constexpr int REPEATS = 3;

#define HIP_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(_e)); \
} while (0)

#define ROCBLAS_CHECK(call) do { \
    rocblas_status _s = (call); \
    if (_s != rocblas_status_success) \
        throw std::runtime_error("rocBLAS call failed"); \
} while (0)

#define LT_CHECK(call) do { \
    hipblasStatus_t _s = (call); \
    if (_s != HIPBLAS_STATUS_SUCCESS) \
        throw std::runtime_error("hipBLASLt call failed"); \
} while (0)

enum class Pattern {
    SINGLE_TERM,
    TWO_TERM,
    POSITIVE,
    RANDOM_SIGN,
    ADJACENT_CANCEL,
    BLOCK_CANCEL,
    MULTISCALE_BLOCK_CANCEL
};

const char* pattern_name(Pattern p) {
    switch (p) {
        case Pattern::SINGLE_TERM: return "SINGLE_TERM";
        case Pattern::TWO_TERM: return "TWO_TERM";
        case Pattern::POSITIVE: return "POSITIVE";
        case Pattern::RANDOM_SIGN: return "RANDOM_SIGN";
        case Pattern::ADJACENT_CANCEL: return "ADJACENT_CANCEL";
        case Pattern::BLOCK_CANCEL: return "BLOCK_CANCEL";
        case Pattern::MULTISCALE_BLOCK_CANCEL: return "MULTISCALE_BLOCK_CANCEL";
    }
    return "UNKNOWN";
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Exactly representable dyadic value in roughly [0.5, 1.0).
float dyadic_value(std::uint64_t index, std::uint64_t row) {
    const std::uint64_t h = splitmix64(index ^ (row * 0xD1B54A32D192ED03ull) ^ SEED);
    const std::uint32_t q = static_cast<std::uint32_t>((h >> 53) & 1023u);
    return 0.5f + static_cast<float>(q) * (1.0f / 2048.0f);
}

float signed_random_value(std::uint64_t index, std::uint64_t row) {
    const std::uint64_t h = splitmix64(index ^ (row * 0x94D049BB133111EBull) ^ SEED);
    const float mag = 0.5f + static_cast<float>((h >> 53) & 1023u) * (1.0f / 2048.0f);
    return (h & 1u) ? mag : -mag;
}

struct DeviceBuffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { reset(n); }
    ~DeviceBuffer() { if (ptr) (void)hipFree(ptr); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void reset(std::size_t n) {
        if (ptr) {
            (void)hipFree(ptr);
            ptr = nullptr;
        }
        bytes = n;
        if (bytes) HIP_CHECK(hipMalloc(&ptr, bytes));
    }

    float* f32() { return static_cast<float*>(ptr); }
    const float* f32() const { return static_cast<const float*>(ptr); }
};

struct Metrics {
    bool pass = false;
    int failures = 0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double max_rel = 0.0;
    double eta_over_u = 0.0;
    bool bitwise_same = true;
};

struct CaseData {
    std::vector<float> A;
    std::vector<float> x;
    std::vector<double> ref;
    std::vector<double> sum_abs_terms;
};

void fill_common_x(CaseData& c, std::int64_t n) {
    c.x.assign(static_cast<std::size_t>(n), 1.0f);
}

void finalize_reference(CaseData& c, std::int64_t n) {
    c.ref.assign(static_cast<std::size_t>(M), 0.0);
    c.sum_abs_terms.assign(static_cast<std::size_t>(M), 0.0);
    for (std::int64_t r = 0; r < M; ++r) {
        long double s = 0.0L;
        long double sa = 0.0L;
        const std::size_t base = static_cast<std::size_t>(r * n);
        for (std::int64_t k = 0; k < n; ++k) {
            const long double t =
                static_cast<long double>(c.A[base + static_cast<std::size_t>(k)]) *
                static_cast<long double>(c.x[static_cast<std::size_t>(k)]);
            s += t;
            sa += std::fabs(t);
        }
        c.ref[static_cast<std::size_t>(r)] = static_cast<double>(s);
        c.sum_abs_terms[static_cast<std::size_t>(r)] = static_cast<double>(sa);
    }
}

CaseData make_case(Pattern p, std::int64_t n) {
    CaseData c;
    c.A.assign(static_cast<std::size_t>(M * n), 0.0f);
    fill_common_x(c, n);

    for (std::int64_t r = 0; r < M; ++r) {
        float* row = c.A.data() + static_cast<std::size_t>(r * n);

        if (p == Pattern::SINGLE_TERM) {
            const std::int64_t k = (r * 131 + 17) % n;
            row[k] = dyadic_value(static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(r));
        } else if (p == Pattern::TWO_TERM) {
            const std::int64_t k0 = (r * 131 + 17) % n;
            const std::int64_t k1 = (k0 + std::max<std::int64_t>(1, n / 3)) % n;
            row[k0] = dyadic_value(static_cast<std::uint64_t>(k0), static_cast<std::uint64_t>(r));
            row[k1] = -0.5f * dyadic_value(static_cast<std::uint64_t>(k1), static_cast<std::uint64_t>(r));
        } else if (p == Pattern::POSITIVE) {
            const float scale = 1.0f / static_cast<float>(n);
            for (std::int64_t k = 0; k < n; ++k)
                row[k] = dyadic_value(static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(r)) * scale;
        } else if (p == Pattern::RANDOM_SIGN) {
            const float scale = 1.0f / std::sqrt(static_cast<float>(n));
            for (std::int64_t k = 0; k < n; ++k)
                row[k] = signed_random_value(static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(r)) * scale;
        } else if (p == Pattern::ADJACENT_CANCEL) {
            for (std::int64_t k = 0; k + 1 < n; k += 2) {
                const float v = dyadic_value(static_cast<std::uint64_t>(k / 2), static_cast<std::uint64_t>(r));
                row[k] = v;
                row[k + 1] = -v;
            }
        } else if (p == Pattern::BLOCK_CANCEL) {
            const std::int64_t half = n / 2;
            for (std::int64_t k = 0; k < half; ++k) {
                const float v = dyadic_value(static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(r));
                row[k] = v;
                row[k + half] = -v;
            }
        } else if (p == Pattern::MULTISCALE_BLOCK_CANCEL) {
            // Four scales, each represented by a positive block and a matching negative block.
            const std::array<float, 4> scales = {{1.0f, 0.125f, 0.015625f, 0.001953125f}};
            const std::int64_t chunk = n / 8;
            for (int s = 0; s < 4; ++s) {
                const std::int64_t pos0 = static_cast<std::int64_t>(2 * s) * chunk;
                const std::int64_t neg0 = pos0 + chunk;
                for (std::int64_t j = 0; j < chunk; ++j) {
                    const float v =
                        dyadic_value(static_cast<std::uint64_t>(j), static_cast<std::uint64_t>(r + 97 * s)) *
                        scales[static_cast<std::size_t>(s)];
                    row[pos0 + j] = v;
                    row[neg0 + j] = -v;
                }
            }
        }
    }

    finalize_reference(c, n);
    return c;
}

CaseData make_distance_case(std::int64_t n, std::int64_t distance) {
    CaseData c;
    c.A.assign(static_cast<std::size_t>(M * n), 0.0f);
    fill_common_x(c, n);

    for (std::int64_t r = 0; r < M; ++r) {
        float* row = c.A.data() + static_cast<std::size_t>(r * n);
        const std::int64_t group = 2 * distance;
        std::uint64_t pair_id = 0;
        for (std::int64_t base = 0; base + distance < n; base += group) {
            const std::int64_t count = std::min<std::int64_t>(distance, n - (base + distance));
            for (std::int64_t j = 0; j < count; ++j, ++pair_id) {
                const float v = dyadic_value(pair_id, static_cast<std::uint64_t>(r));
                row[base + j] = v;
                row[base + distance + j] = -v;
            }
        }
    }

    finalize_reference(c, n);
    return c;
}

Metrics compare(const std::vector<float>& got,
                const std::vector<double>& ref,
                const std::vector<double>& sum_abs_terms) {
    Metrics m;
    long double total_abs = 0.0L;

    for (std::size_t i = 0; i < got.size(); ++i) {
        const double e = static_cast<double>(got[i]) - ref[i];
        const double ae = std::fabs(e);
        const double rel = ae / std::max(std::fabs(ref[i]), REL_EPS);
        const double tol = ATOL + RTOL * std::fabs(ref[i]);

        if (ae > tol) ++m.failures;
        m.max_abs = std::max(m.max_abs, ae);
        m.max_rel = std::max(m.max_rel, rel);
        total_abs += ae;

        const double denom = sum_abs_terms[i];
        if (denom > 0.0) {
            const double eta_u = (ae / denom) / U32;
            m.eta_over_u = std::max(m.eta_over_u, eta_u);
        }
    }

    m.mean_abs = static_cast<double>(total_abs / static_cast<long double>(got.size()));
    m.pass = (m.failures == 0);
    return m;
}

class RocblasRunner {
public:
    explicit RocblasRunner(hipStream_t stream) {
        ROCBLAS_CHECK(rocblas_create_handle(&handle_));
        ROCBLAS_CHECK(rocblas_set_stream(handle_, stream));
        ROCBLAS_CHECK(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host));
    }

    ~RocblasRunner() {
        if (handle_) (void)rocblas_destroy_handle(handle_);
    }

    void run(const float* A, const float* x, float* y,
             std::int64_t m, std::int64_t n, std::int64_t lda) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        ROCBLAS_CHECK(
            rocblas_sgemv(handle_, rocblas_operation_transpose,
                          static_cast<rocblas_int>(n),
                          static_cast<rocblas_int>(m),
                          &alpha, A, static_cast<rocblas_int>(lda),
                          x, 1, &beta, y, 1));
    }

private:
    rocblas_handle handle_ = nullptr;
};

struct AlgoChoice {
    hipblasLtMatmulAlgo_t algo{};
    int index = -1;
    std::size_t workspace = 0;
    int rank = -1;
};

class LtRunner {
public:
    LtRunner(std::int64_t m, std::int64_t n, std::int64_t lda,
             std::uint64_t max_workspace)
        : m_(m), n_(n), lda_(lda), max_workspace_(max_workspace) {
        LT_CHECK(hipblasLtCreate(&handle_));
        LT_CHECK(hipblasLtMatmulDescCreate(&op_, HIPBLAS_COMPUTE_32F, HIP_R_32F));

        hipblasOperation_t trans_a = HIPBLAS_OP_T;
        hipblasOperation_t trans_b = HIPBLAS_OP_N;
        LT_CHECK(hipblasLtMatmulDescSetAttribute(
            op_, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
        LT_CHECK(hipblasLtMatmulDescSetAttribute(
            op_, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

        LT_CHECK(hipblasLtMatrixLayoutCreate(
            &a_, HIP_R_32F, static_cast<std::uint64_t>(n_),
            static_cast<std::uint64_t>(m_), lda_));
        LT_CHECK(hipblasLtMatrixLayoutCreate(
            &b_, HIP_R_32F, static_cast<std::uint64_t>(n_), 1,
            static_cast<std::int64_t>(n_)));
        LT_CHECK(hipblasLtMatrixLayoutCreate(
            &c_, HIP_R_32F, static_cast<std::uint64_t>(m_), 1,
            static_cast<std::int64_t>(m_)));
        LT_CHECK(hipblasLtMatrixLayoutCreate(
            &d_, HIP_R_32F, static_cast<std::uint64_t>(m_), 1,
            static_cast<std::int64_t>(m_)));

        LT_CHECK(hipblasLtMatmulPreferenceCreate(&pref_));
        const std::uint64_t limit = max_workspace_;
        LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
            pref_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &limit, sizeof(limit)));
    }

    ~LtRunner() {
        if (workspace_) (void)hipFree(workspace_);
        if (pref_) (void)hipblasLtMatmulPreferenceDestroy(pref_);
        if (d_) (void)hipblasLtMatrixLayoutDestroy(d_);
        if (c_) (void)hipblasLtMatrixLayoutDestroy(c_);
        if (b_) (void)hipblasLtMatrixLayoutDestroy(b_);
        if (a_) (void)hipblasLtMatrixLayoutDestroy(a_);
        if (op_) (void)hipblasLtMatmulDescDestroy(op_);
        if (handle_) (void)hipblasLtDestroy(handle_);
    }

    bool load_solution(int solution_index) {
        std::vector<int> indices(1, solution_index);
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        if (hipblaslt_ext::getAlgosFromIndex(handle_, indices, results) != HIPBLAS_STATUS_SUCCESS)
            return false;

        for (std::size_t i = 0; i < results.size(); ++i) {
            hipblasLtMatmulHeuristicResult_t& r = results[i];

            // ROCm 7.2.x getIndexFromAlgo takes a non-const reference.
            // Keep this object non-const for C++17/ROCm compatibility.
            const int idx = hipblaslt_ext::getIndexFromAlgo(r.algo);
            if (idx != solution_index) continue;

            const float alpha = 1.0f;
            const float beta = 0.0f;
            std::size_t ws = 0;
            if (hipblaslt_ext::matmulIsAlgoSupported(
                    handle_, op_, &alpha, a_, b_, &beta, c_, d_, r.algo, ws)
                != HIPBLAS_STATUS_SUCCESS)
                continue;
            if (ws > max_workspace_) continue;

            AlgoChoice choice;
            choice.algo = r.algo;
            choice.index = idx;
            choice.workspace = ws;
            choice.rank = -1;
            set_choice(choice);
            return true;
        }
        return false;
    }

    std::vector<AlgoChoice> heuristic_choices(int count) {
        std::vector<hipblasLtMatmulHeuristicResult_t> h(
            static_cast<std::size_t>(count));
        int returned = 0;
        const hipblasStatus_t s = hipblasLtMatmulAlgoGetHeuristic(
            handle_, op_, a_, b_, c_, d_, pref_,
            count, h.data(), &returned);
        if (s != HIPBLAS_STATUS_SUCCESS || returned <= 0)
            return {};

        std::vector<AlgoChoice> out;
        for (int i = 0; i < returned; ++i) {
            hipblasLtMatmulHeuristicResult_t& r = h[static_cast<std::size_t>(i)];
            if (r.state != HIPBLAS_STATUS_SUCCESS) continue;
            if (r.workspaceSize > max_workspace_) continue;

            AlgoChoice choice;
            choice.algo = r.algo;
            choice.index = hipblaslt_ext::getIndexFromAlgo(r.algo);
            choice.workspace = r.workspaceSize;
            choice.rank = i;
            out.push_back(choice);
        }
        return out;
    }

    void set_choice(const AlgoChoice& choice) {
        ensure_workspace(choice.workspace);
        selected_ = choice;
        has_selected_ = true;
    }

    bool select_default(int preferred_solution = 1221) {
        if (load_solution(preferred_solution)) return true;
        std::vector<AlgoChoice> h = heuristic_choices(1);
        if (h.empty()) return false;
        set_choice(h.front());
        return true;
    }

    int solution_index() const { return has_selected_ ? selected_.index : -1; }
    std::size_t workspace_bytes() const { return has_selected_ ? selected_.workspace : 0; }

    void run(const float* A, const float* x, float* y, hipStream_t stream) {
        if (!has_selected_) throw std::runtime_error("hipBLASLt: no selected algorithm");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        LT_CHECK(hipblasLtMatmul(
            handle_, op_, &alpha,
            A, a_, x, b_, &beta,
            y, c_, y, d_,
            &selected_.algo,
            selected_.workspace ? workspace_ : nullptr,
            selected_.workspace, stream));
    }

private:
    void ensure_workspace(std::size_t bytes) {
        if (bytes <= workspace_capacity_) return;
        if (workspace_) {
            (void)hipFree(workspace_);
            workspace_ = nullptr;
            workspace_capacity_ = 0;
        }
        if (bytes) {
            HIP_CHECK(hipMalloc(&workspace_, bytes));
            workspace_capacity_ = bytes;
        }
    }

    std::int64_t m_, n_, lda_;
    std::uint64_t max_workspace_;
    hipblasLtHandle_t handle_ = nullptr;
    hipblasLtMatmulDesc_t op_ = nullptr;
    hipblasLtMatrixLayout_t a_ = nullptr;
    hipblasLtMatrixLayout_t b_ = nullptr;
    hipblasLtMatrixLayout_t c_ = nullptr;
    hipblasLtMatrixLayout_t d_ = nullptr;
    hipblasLtMatmulPreference_t pref_ = nullptr;
    void* workspace_ = nullptr;
    std::size_t workspace_capacity_ = 0;
    AlgoChoice selected_{};
    bool has_selected_ = false;
};

struct GPUCase {
    std::int64_t n = 0;
    DeviceBuffer A;
    DeviceBuffer x;
    DeviceBuffer y;
    DeviceBuffer native_workspace;

    explicit GPUCase(std::int64_t n_)
        : n(n_),
          A(static_cast<std::size_t>(M * n) * sizeof(float)),
          x(static_cast<std::size_t>(n) * sizeof(float)),
          y(static_cast<std::size_t>(M) * sizeof(float)),
          native_workspace(static_cast<std::size_t>(gemv_workspace_bytes(M, n))) {}
};

void reset_native_workspace(GPUCase& g, hipStream_t stream) {
    if (g.native_workspace.bytes != 0) {
        HIP_CHECK(hipMemsetAsync(
            g.native_workspace.ptr, 0, g.native_workspace.bytes, stream));
    }
}

void upload_case(GPUCase& g, const CaseData& c, hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(
        g.A.ptr, c.A.data(), c.A.size() * sizeof(float),
        hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(
        g.x.ptr, c.x.data(), c.x.size() * sizeof(float),
        hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

std::vector<float> copy_y(GPUCase& g, hipStream_t stream) {
    std::vector<float> h(static_cast<std::size_t>(M));
    HIP_CHECK(hipMemcpyAsync(
        h.data(), g.y.ptr, h.size() * sizeof(float),
        hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return h;
}

template <class Launch>
Metrics run_repeated(GPUCase& g,
                     const CaseData& c,
                     hipStream_t stream,
                     Launch launch) {
    std::vector<float> first;
    Metrics metrics;

    for (int rep = 0; rep < REPEATS; ++rep) {
        HIP_CHECK(hipMemsetAsync(g.y.ptr, 0, static_cast<std::size_t>(M) * sizeof(float), stream));
        launch();
        HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<float> got = copy_y(g, stream);

        if (rep == 0) {
            first = got;
            metrics = compare(got, c.ref, c.sum_abs_terms);
        } else if (std::memcmp(first.data(), got.data(), first.size() * sizeof(float)) != 0) {
            metrics.bitwise_same = false;
        }
    }
    return metrics;
}

void write_csv_header(std::ofstream& f, bool with_n, bool with_pattern) {
    if (with_pattern) f << "pattern,";
    if (with_n) f << "N,";
    f << "backend,status,failures,max_abs,mean_abs,max_rel,eta_over_u,repeat,solution,workspace,path\n";
}

void write_metrics_row(std::ofstream& f,
                       const std::string& prefix,
                       const char* backend,
                       const Metrics& m,
                       int solution,
                       std::uint64_t workspace,
                       const char* path) {
    f << prefix
      << backend << ","
      << (m.pass ? "PASS" : "FAIL") << ","
      << m.failures << ","
      << std::setprecision(17) << m.max_abs << ","
      << m.mean_abs << ","
      << m.max_rel << ","
      << m.eta_over_u << ","
      << (m.bitwise_same ? "BITWISE_SAME" : "DIFFERENT") << ","
      << solution << ","
      << workspace << ","
      << (path ? path : "") << "\n";
}

void print_line(const char* pattern,
                std::int64_t n,
                const char* backend,
                const Metrics& m,
                int solution,
                std::uint64_t workspace,
                const char* path) {
    std::cout << std::left << std::setw(24) << pattern
              << " N=" << std::setw(6) << n
              << " " << std::setw(10) << backend
              << " " << std::setw(5) << (m.pass ? "PASS" : "FAIL")
              << " fail=" << std::setw(3) << m.failures
              << " max_abs=" << std::scientific << std::setprecision(3) << m.max_abs
              << " eta/u=" << m.eta_over_u
              << " repeat=" << (m.bitwise_same ? "BITWISE_SAME" : "DIFFERENT");
    if (solution >= 0) std::cout << " sol=" << solution;
    if (workspace || std::string(backend) == "native" || std::string(backend) == "hipBLASLt")
        std::cout << " ws=" << workspace;
    if (path && *path) std::cout << " path=" << path;
    std::cout << "\n";
}

void baseline(hipStream_t stream,
              const std::string& out_dir,
              int preferred_solution) {
    const std::array<std::int64_t, 5> ns = {{256, 1024, 4096, 16384, 65536}};
    const std::array<Pattern, 7> patterns = {{
        Pattern::SINGLE_TERM,
        Pattern::TWO_TERM,
        Pattern::POSITIVE,
        Pattern::RANDOM_SIGN,
        Pattern::ADJACENT_CANCEL,
        Pattern::BLOCK_CANCEL,
        Pattern::MULTISCALE_BLOCK_CANCEL
    }};

    std::ofstream csv(out_dir + "/accuracy_baseline.csv");
    write_csv_header(csv, true, true);

    std::cout << "FP32 numerical baseline: native vs rocBLAS vs hipBLASLt\n";
    std::cout << "M=" << M << " seed=" << SEED
              << " atol=" << ATOL << " rtol=" << RTOL
              << " u32=" << std::scientific << U32 << "\n\n";

    RocblasRunner rb(stream);

    for (std::size_t ni = 0; ni < ns.size(); ++ni) {
        const std::int64_t n = ns[ni];
        GPUCase g(n);
        LtRunner lt(M, n, n, LT_MAX_WORKSPACE);
        if (!lt.select_default(preferred_solution))
            throw std::runtime_error("hipBLASLt: no usable algorithm");

        const char* path = gemv_path_name(M, n);
        const std::uint64_t native_ws = gemv_workspace_bytes(M, n);

        for (std::size_t pi = 0; pi < patterns.size(); ++pi) {
            const Pattern p = patterns[pi];
            CaseData c = make_case(p, n);
            upload_case(g, c, stream);

            Metrics mn = run_repeated(g, c, stream, [&]() {
                reset_native_workspace(g, stream);
                HIP_CHECK(gemv(g.A.f32(), g.x.f32(), g.y.f32(),
                               M, n, n, g.native_workspace.ptr, stream));
            });
            print_line(pattern_name(p), n, "native", mn, -1, native_ws, path);
            write_metrics_row(csv,
                std::string(pattern_name(p)) + "," + std::to_string(n) + ",",
                "native", mn, -1, native_ws, path);

            Metrics mr = run_repeated(g, c, stream, [&]() {
                rb.run(g.A.f32(), g.x.f32(), g.y.f32(), M, n, n);
            });
            print_line(pattern_name(p), n, "rocBLAS", mr, -1, 0, "");
            write_metrics_row(csv,
                std::string(pattern_name(p)) + "," + std::to_string(n) + ",",
                "rocBLAS", mr, -1, 0, "");

            Metrics ml = run_repeated(g, c, stream, [&]() {
                lt.run(g.A.f32(), g.x.f32(), g.y.f32(), stream);
            });
            print_line(pattern_name(p), n, "hipBLASLt", ml,
                       lt.solution_index(), lt.workspace_bytes(), "");
            write_metrics_row(csv,
                std::string(pattern_name(p)) + "," + std::to_string(n) + ",",
                "hipBLASLt", ml, lt.solution_index(),
                static_cast<std::uint64_t>(lt.workspace_bytes()), "");
        }
    }
}


struct DistanceResult {
    std::int64_t distance = 0;
    Metrics native;
    Metrics rocblas;
    Metrics hipblaslt;
};

struct TransitionEvent {
    std::string sweep;
    std::string backend;
    std::int64_t last_zero_distance = -1;
    std::int64_t first_nonzero_distance = -1;
    double first_nonzero_max_abs = 0.0;
    double first_nonzero_eta_over_u = 0.0;
};

bool is_exact_zero(const Metrics& m) {
    return m.max_abs == 0.0;
}

DistanceResult measure_distance_case(std::int64_t distance,
                                     GPUCase& g,
                                     RocblasRunner& rb,
                                     LtRunner& lt,
                                     hipStream_t stream) {
    const std::int64_t n = g.n;
    CaseData c = make_distance_case(n, distance);
    upload_case(g, c, stream);

    DistanceResult r;
    r.distance = distance;
    r.native = run_repeated(g, c, stream, [&]() {
        reset_native_workspace(g, stream);
        HIP_CHECK(gemv(g.A.f32(), g.x.f32(), g.y.f32(),
                       M, n, n, g.native_workspace.ptr, stream));
    });
    r.rocblas = run_repeated(g, c, stream, [&]() {
        rb.run(g.A.f32(), g.x.f32(), g.y.f32(), M, n, n);
    });
    r.hipblaslt = run_repeated(g, c, stream, [&]() {
        lt.run(g.A.f32(), g.x.f32(), g.y.f32(), stream);
    });
    return r;
}

const Metrics& distance_metrics(const DistanceResult& r, int backend) {
    if (backend == 0) return r.native;
    if (backend == 1) return r.rocblas;
    return r.hipblaslt;
}

const char* distance_backend_name(int backend) {
    if (backend == 0) return "native";
    if (backend == 1) return "rocBLAS";
    return "hipBLASLt";
}

std::vector<TransitionEvent> detect_zero_to_nonzero(
        const std::string& sweep,
        const std::vector<DistanceResult>& rows) {
    std::vector<TransitionEvent> events;
    for (int backend = 0; backend < 3; ++backend) {
        bool have_prev = false;
        bool prev_zero = false;
        std::int64_t prev_distance = -1;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const Metrics& m = distance_metrics(rows[i], backend);
            const bool now_zero = is_exact_zero(m);
            if (have_prev && prev_zero && !now_zero) {
                TransitionEvent e;
                e.sweep = sweep;
                e.backend = distance_backend_name(backend);
                e.last_zero_distance = prev_distance;
                e.first_nonzero_distance = rows[i].distance;
                e.first_nonzero_max_abs = m.max_abs;
                e.first_nonzero_eta_over_u = m.eta_over_u;
                events.push_back(e);
            }
            have_prev = true;
            prev_zero = now_zero;
            prev_distance = rows[i].distance;
        }
    }
    return events;
}

bool is_transition_at(const std::vector<TransitionEvent>& events,
                      const char* backend,
                      std::int64_t distance) {
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].backend == backend &&
            events[i].first_nonzero_distance == distance)
            return true;
    }
    return false;
}

void print_distance_row(const DistanceResult& r,
                        const std::vector<TransitionEvent>& events) {
    std::cout << "distance=" << std::left << std::setw(6) << r.distance;
    for (int backend = 0; backend < 3; ++backend) {
        const char* name = distance_backend_name(backend);
        const Metrics& m = distance_metrics(r, backend);
        std::cout << " " << std::setw(10) << name
                  << " " << std::setw(5) << (m.pass ? "PASS" : "FAIL")
                  << " max_abs=" << std::scientific << std::setprecision(3)
                  << m.max_abs
                  << " eta/u=" << m.eta_over_u;
        if (is_transition_at(events, name, r.distance))
            std::cout << " [ZERO->NONZERO]";
    }
    std::cout << "\n";
}

void write_distance_csv(const std::string& filename,
                        const std::vector<DistanceResult>& rows,
                        const std::vector<TransitionEvent>& events,
                        int lt_solution,
                        std::uint64_t lt_workspace,
                        std::uint64_t native_workspace,
                        const char* native_path,
                        const std::map<std::int64_t, std::string>* phases = nullptr) {
    std::ofstream csv(filename);
    csv << "distance,phase,backend,status,failures,max_abs,mean_abs,max_rel,"
           "eta_over_u,repeat,solution,workspace,path,exact_zero,zero_to_nonzero\n";

    for (std::size_t i = 0; i < rows.size(); ++i) {
        const DistanceResult& r = rows[i];
        std::string phase = "FULL";
        if (phases) {
            std::map<std::int64_t, std::string>::const_iterator it = phases->find(r.distance);
            if (it != phases->end()) phase = it->second;
        }

        for (int backend = 0; backend < 3; ++backend) {
            const char* name = distance_backend_name(backend);
            const Metrics& m = distance_metrics(r, backend);
            const int sol = (backend == 2) ? lt_solution : -1;
            const std::uint64_t ws =
                (backend == 0) ? native_workspace :
                (backend == 2 ? lt_workspace : 0);
            const char* path = (backend == 0) ? native_path : "";

            csv << r.distance << "," << phase << "," << name << ","
                << (m.pass ? "PASS" : "FAIL") << ","
                << m.failures << ","
                << std::setprecision(17) << m.max_abs << ","
                << m.mean_abs << ","
                << m.max_rel << ","
                << m.eta_over_u << ","
                << (m.bitwise_same ? "BITWISE_SAME" : "DIFFERENT") << ","
                << sol << "," << ws << "," << path << ","
                << (is_exact_zero(m) ? 1 : 0) << ","
                << (is_transition_at(events, name, r.distance) ? 1 : 0)
                << "\n";
        }
    }
}

void write_transition_csv(const std::string& filename,
                          const std::vector<TransitionEvent>& events) {
    std::ofstream csv(filename);
    csv << "sweep,backend,last_zero_distance,first_nonzero_distance,"
           "first_nonzero_max_abs,first_nonzero_eta_over_u\n";
    for (std::size_t i = 0; i < events.size(); ++i) {
        const TransitionEvent& e = events[i];
        csv << e.sweep << "," << e.backend << ","
            << e.last_zero_distance << ","
            << e.first_nonzero_distance << ","
            << std::setprecision(17) << e.first_nonzero_max_abs << ","
            << e.first_nonzero_eta_over_u << "\n";
    }
}

void print_transitions(const std::string& title,
                       const std::vector<TransitionEvent>& events) {
    std::cout << title << "\n";
    if (events.empty()) {
        std::cout << "  no exact zero->nonzero transition observed\n";
        return;
    }
    for (std::size_t i = 0; i < events.size(); ++i) {
        const TransitionEvent& e = events[i];
        std::cout << "  " << std::setw(10) << e.backend
                  << " " << e.last_zero_distance
                  << " -> " << e.first_nonzero_distance
                  << "  max_abs=" << std::scientific << std::setprecision(3)
                  << e.first_nonzero_max_abs
                  << " eta/u=" << e.first_nonzero_eta_over_u << "\n";
    }
}

void tomography(hipStream_t stream,
                const std::string& out_dir,
                int preferred_solution,
                std::vector<TransitionEvent>& all_events) {
    const std::int64_t n = 65536;
    const std::array<std::int64_t, 16> distances = {{
        1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768
    }};

    GPUCase g(n);
    RocblasRunner rb(stream);
    LtRunner lt(M, n, n, LT_MAX_WORKSPACE);
    if (!lt.select_default(preferred_solution))
        throw std::runtime_error("hipBLASLt: no usable algorithm");

    const char* path = gemv_path_name(M, n);
    const std::uint64_t native_ws = gemv_workspace_bytes(M, n);

    std::vector<DistanceResult> rows;
    for (std::size_t i = 0; i < distances.size(); ++i)
        rows.push_back(measure_distance_case(distances[i], g, rb, lt, stream));

    const std::vector<TransitionEvent> events =
        detect_zero_to_nonzero("full_tomography", rows);
    all_events.insert(all_events.end(), events.begin(), events.end());

    std::cout << "\nCancellation-distance tomography, N=" << n << "\n";
    for (std::size_t i = 0; i < rows.size(); ++i)
        print_distance_row(rows[i], events);
    print_transitions("Exact zero->nonzero transitions:", events);

    write_distance_csv(
        out_dir + "/cancellation_distance.csv",
        rows, events, lt.solution_index(),
        static_cast<std::uint64_t>(lt.workspace_bytes()),
        native_ws, path);
}

std::pair<std::int64_t, std::int64_t> hipblaslt_zero_nonzero_bracket(
        const std::map<std::int64_t, DistanceResult>& results) {
    bool have_prev = false;
    bool prev_zero = false;
    std::int64_t prev_distance = -1;
    for (std::map<std::int64_t, DistanceResult>::const_iterator it = results.begin();
         it != results.end(); ++it) {
        const bool now_zero = is_exact_zero(it->second.hipblaslt);
        if (have_prev && prev_zero && !now_zero)
            return std::make_pair(prev_distance, it->first);
        have_prev = true;
        prev_zero = now_zero;
        prev_distance = it->first;
    }
    return std::make_pair(-1, -1);
}

void add_zoom_range(std::int64_t lo,
                    std::int64_t hi,
                    std::int64_t step,
                    const std::string& phase,
                    std::map<std::int64_t, DistanceResult>& results,
                    std::map<std::int64_t, std::string>& phases,
                    GPUCase& g,
                    RocblasRunner& rb,
                    LtRunner& lt,
                    hipStream_t stream) {
    if (lo < 0 || hi < 0 || hi <= lo || step <= 0) return;
    for (std::int64_t d = lo + step; d < hi; d += step) {
        if (results.find(d) == results.end()) {
            results[d] = measure_distance_case(d, g, rb, lt, stream);
            phases[d] = phase;
        }
    }
}

void breakpoint_zoom(hipStream_t stream,
                     const std::string& out_dir,
                     int preferred_solution,
                     std::vector<TransitionEvent>& all_events) {
    const std::int64_t n = 65536;
    const std::int64_t lo = 8192;
    const std::int64_t hi = 16384;

    GPUCase g(n);
    RocblasRunner rb(stream);
    LtRunner lt(M, n, n, LT_MAX_WORKSPACE);
    if (!lt.select_default(preferred_solution))
        throw std::runtime_error("hipBLASLt: no usable algorithm");

    const char* path = gemv_path_name(M, n);
    const std::uint64_t native_ws = gemv_workspace_bytes(M, n);

    std::map<std::int64_t, DistanceResult> results;
    std::map<std::int64_t, std::string> phases;

    // Stage 1: coarse sweep, 512-element resolution.
    for (std::int64_t d = lo; d <= hi; d += 512) {
        results[d] = measure_distance_case(d, g, rb, lt, stream);
        phases[d] = "COARSE512";
    }

    // Stage 2: zoom the first hipBLASLt exact-zero -> nonzero bracket to 64.
    std::pair<std::int64_t, std::int64_t> bracket =
        hipblaslt_zero_nonzero_bracket(results);
    if (bracket.first >= 0)
        add_zoom_range(bracket.first, bracket.second, 64, "FINE64",
                       results, phases, g, rb, lt, stream);

    // Stage 3: zoom the refined bracket to 8 elements.
    bracket = hipblaslt_zero_nonzero_bracket(results);
    if (bracket.first >= 0)
        add_zoom_range(bracket.first, bracket.second, 8, "MICRO8",
                       results, phases, g, rb, lt, stream);

    std::vector<DistanceResult> rows;
    rows.reserve(results.size());
    for (std::map<std::int64_t, DistanceResult>::const_iterator it = results.begin();
         it != results.end(); ++it)
        rows.push_back(it->second);

    const std::vector<TransitionEvent> events =
        detect_zero_to_nonzero("zoom_8k_16k", rows);
    all_events.insert(all_events.end(), events.begin(), events.end());

    std::cout << "\nBreakpoint zoom sweep, N=" << n
              << " distance=[" << lo << "," << hi << "]"
              << " coarse=512 fine=64 micro=8\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::cout << "[" << std::setw(9) << phases[rows[i].distance] << "] ";
        print_distance_row(rows[i], events);
    }
    print_transitions("Zoom zero->nonzero transitions:", events);

    write_distance_csv(
        out_dir + "/cancellation_zoom_8k_16k.csv",
        rows, events, lt.solution_index(),
        static_cast<std::uint64_t>(lt.workspace_bytes()),
        native_ws, path, &phases);
}

void solution_sweep(hipStream_t stream,
                    const std::string& out_dir,
                    int preferred_solution) {
    const std::int64_t n = 65536;
    const std::array<Pattern,3> patterns = {{
        Pattern::POSITIVE,
        Pattern::ADJACENT_CANCEL,
        Pattern::BLOCK_CANCEL
    }};

    GPUCase g(n);
    LtRunner lt(M, n, n, LT_MAX_WORKSPACE);

    std::ofstream csv(out_dir + "/solution_accuracy.csv");
    write_csv_header(csv, true, true);

    // Run the explicitly restored solution first (when available), then
    // the heuristic alternatives returned by the installed hipBLASLt.
    std::vector<AlgoChoice> hs = lt.heuristic_choices(LT_HEURISTIC_COUNT);

    std::cout << "\nhipBLASLt solution-accuracy sweep at N=" << n << "\n";

    for (std::size_t pi = 0; pi < patterns.size(); ++pi) {
        const Pattern p = patterns[pi];
        CaseData c = make_case(p, n);
        upload_case(g, c, stream);

        bool restored_done = false;
        if (lt.load_solution(preferred_solution)) {
            Metrics m = run_repeated(g, c, stream, [&]() {
                lt.run(g.A.f32(), g.x.f32(), g.y.f32(), stream);
            });
            print_line(pattern_name(p), n, "hipBLASLt", m,
                       lt.solution_index(), lt.workspace_bytes(), "");
            write_metrics_row(csv,
                std::string(pattern_name(p)) + "," + std::to_string(n) + ",",
                "hipBLASLt", m, lt.solution_index(),
                static_cast<std::uint64_t>(lt.workspace_bytes()), "");
            restored_done = true;
        }

        for (std::size_t hi = 0; hi < hs.size(); ++hi) {
            lt.set_choice(hs[hi]);
            Metrics m = run_repeated(g, c, stream, [&]() {
                lt.run(g.A.f32(), g.x.f32(), g.y.f32(), stream);
            });

            std::cout << std::left << std::setw(24) << pattern_name(p)
                      << " N=" << n << " hipBLASLt  "
                      << (m.pass ? "PASS" : "FAIL")
                      << " fail=" << m.failures
                      << " max_abs=" << std::scientific << std::setprecision(3) << m.max_abs
                      << " eta/u=" << m.eta_over_u
                      << " repeat=" << (m.bitwise_same ? "BITWISE_SAME" : "DIFFERENT")
                      << " sol=" << hs[hi].index
                      << " rank=" << hs[hi].rank
                      << " ws=" << hs[hi].workspace << "\n";

            write_metrics_row(csv,
                std::string(pattern_name(p)) + "," + std::to_string(n) + ",",
                "hipBLASLt", m, hs[hi].index,
                static_cast<std::uint64_t>(hs[hi].workspace), "");
        }

        if (!restored_done && hs.empty())
            throw std::runtime_error("hipBLASLt: no solution for sweep");
    }
}

void write_report(const std::string& out_dir) {
    std::ofstream f(out_dir + "/numerical_report.md");
    f << "# FP32 GEMV numerical analysis\n\n";
    f << "This report is generated by `numerical/fp32_accuracy.cpp`.\n\n";
    f << "## Method\n\n";
    f << "- FP64 host reference using the exact FP32 operands.\n";
    f << "- Tolerance: `atol=1e-4`, `rtol=1e-4`.\n";
    f << "- `eta/u = (|error| / sum |a_k x_k|) / 2^-24`.\n";
    f << "- Three repeated launches are compared bitwise.\n";
    f << "- Backends: native gfx1100 GEMV, rocBLAS SGEMV, hipBLASLt GEMM-as-GEMV.\n\n";
    f << "## Experiments\n\n";
    f << "1. Baseline patterns across N = 256, 1024, 4096, 16384, 65536.\n";
    f << "2. Cancellation-distance tomography at N = 65536 with exact zero->nonzero detection.\n";
    f << "3. Adaptive 8K-16K breakpoint zoom: 512 -> 64 -> 8 element resolution.\n";
    f << "4. hipBLASLt restored-solution and heuristic accuracy sweep.\n";
    f << "5. Optional ISA capture and topology correlation via `tools/`.\n\n";
    f << "Native workspace is explicitly zero-initialized before every repeated launch.\n\n";
    f << "The CSV files in this directory are the authoritative machine-readable results.\n";
}

void ensure_directory(const std::string& path) {
    std::string command = "mkdir -p \"" + path + "\"";
    const int rc = std::system(command.c_str());
    if (rc != 0) throw std::runtime_error("failed to create output directory");
}

struct Args {
    bool isa_target = false;
    int solution = 1221;
    std::int64_t isa_n = 65536;
    std::int64_t isa_distance = 1024;
    int isa_launches = 8;
    std::string out_dir = "numerical_results";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--isa-target") {
            a.isa_target = true;
        } else if (s == "--solution" && i + 1 < argc) {
            a.solution = std::stoi(argv[++i]);
        } else if (s == "--isa-n" && i + 1 < argc) {
            a.isa_n = std::stoll(argv[++i]);
        } else if (s == "--isa-distance" && i + 1 < argc) {
            a.isa_distance = std::stoll(argv[++i]);
        } else if (s == "--isa-launches" && i + 1 < argc) {
            a.isa_launches = std::stoi(argv[++i]);
        } else if (s == "--out-dir" && i + 1 < argc) {
            a.out_dir = argv[++i];
        } else if (s == "--help") {
            std::cout
                << "fp32_accuracy [--out-dir DIR]\\n"
                << "fp32_accuracy --isa-target --solution 1221 "
                << "--isa-n 65536 --isa-distance 1024 --isa-launches 8\\n";
            std::exit(0);
        }
    }
    return a;
}

void run_isa_target(hipStream_t stream, const Args& args) {
    if (args.isa_n <= 0 || args.isa_distance <= 0 ||
        args.isa_distance > args.isa_n / 2 || args.isa_launches <= 0)
        throw std::runtime_error("invalid ISA-target arguments");

    const std::int64_t n = args.isa_n;
    CaseData c = make_distance_case(n, args.isa_distance);
    GPUCase g(n);
    upload_case(g, c, stream);

    LtRunner lt(M, n, n, LT_MAX_WORKSPACE);
    if (!lt.load_solution(args.solution)) {
        if (!lt.select_default(args.solution))
            throw std::runtime_error("unable to load requested or fallback hipBLASLt solution");
    }

    std::cout << "ISA_TARGET_BEGIN solution=" << lt.solution_index()
              << " variant=" << (lt.solution_index() == args.solution ? "restored" : "fallback")
              << " workspace=" << lt.workspace_bytes()
              << " M=" << M
              << " N=" << n
              << " distance=" << args.isa_distance << "\n";

    for (int i = 0; i < args.isa_launches; ++i)
        lt.run(g.A.f32(), g.x.f32(), g.y.f32(), stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::cout << "ISA_TARGET_END launches=" << args.isa_launches << "\n";
}

} // namespace

int main(int argc, char** argv) {
    hipStream_t stream = nullptr;
    try {
        const Args args = parse_args(argc, argv);
        HIP_CHECK(hipStreamCreate(&stream));

        if (args.isa_target) {
            run_isa_target(stream, args);
            (void)hipStreamDestroy(stream);
            return 0;
        }

        ensure_directory(args.out_dir);

        baseline(stream, args.out_dir, args.solution);
        std::vector<TransitionEvent> transition_events;
        tomography(stream, args.out_dir, args.solution, transition_events);
        breakpoint_zoom(stream, args.out_dir, args.solution, transition_events);
        write_transition_csv(
            args.out_dir + "/zero_nonzero_transitions.csv",
            transition_events);
        solution_sweep(stream, args.out_dir, args.solution);
        write_report(args.out_dir);

        std::cout << "\nWrote:\n"
                  << "  \"" << args.out_dir << "/accuracy_baseline.csv\"\n"
                  << "  \"" << args.out_dir << "/cancellation_distance.csv\"\n"
                  << "  \"" << args.out_dir << "/cancellation_zoom_8k_16k.csv\"\n"
                  << "  \"" << args.out_dir << "/zero_nonzero_transitions.csv\"\n"
                  << "  \"" << args.out_dir << "/solution_accuracy.csv\"\n"
                  << "  \"" << args.out_dir << "/numerical_report.md\"\n";

        (void)hipStreamDestroy(stream);
        return 0;
    } catch (const std::exception& e) {
        if (stream) (void)hipStreamDestroy(stream);
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
