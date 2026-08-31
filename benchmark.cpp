#include "gemv.h"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <rocblas/rocblas.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" void* amd_rocblas_create(hipStream_t);
extern "C" void amd_rocblas_destroy(void*);
extern "C" int amd_rocblas_sgemv_launch(void*, const float*, const float*, float*,
                                        std::int64_t, std::int64_t, std::int64_t);
extern "C" int amd_rocblas_success_code();

extern "C" void* amd_lt_create(std::int64_t, std::int64_t, std::int64_t, std::uint64_t);
extern "C" void amd_lt_destroy(void*);
extern "C" int amd_lt_tune(void*, const float*, const float*, float*, hipStream_t,
                           int, int, int, int, int*, int*, int*, std::uint64_t*, double*);
extern "C" int amd_lt_launch(void*, const float*, const float*, float*, hipStream_t);
extern "C" int amd_lt_success_code();
extern "C" int amd_lt_load_solution_index(void*, int);

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t RANDOM_SEED = 12345;
constexpr float ALPHA = 1.0f;
constexpr float BETA = 0.0f;
constexpr double ATOL = 1.0e-4;
constexpr double RTOL = 1.0e-4;
constexpr double RELATIVE_EPSILON = 1.0e-12;

constexpr int GPU_PREHEAT_SECONDS = 2;

// Each backend is measured in an isolated steady-state batch.  The old
// per-sample backend interleave allowed a very slow hipBLASLt launch to
// perturb the immediately following native sample on cache-resident shapes.
constexpr int WARMUP_LAUNCHES = 8;          // per backend, per timing round
constexpr int TIMING_ROUNDS = 3;
constexpr int TIMING_SAMPLES_PER_ROUND = 17;
constexpr int TIMING_SAMPLES =
    TIMING_ROUNDS * TIMING_SAMPLES_PER_ROUND;

constexpr int LT_HEURISTIC_MAX_ALGOS = 32;
constexpr int LT_TUNE_WARMUP = 3;
constexpr int LT_TUNE_SAMPLES = 7;
constexpr int PROFILE_WARMUP_LAUNCHES = 5;
constexpr std::uint64_t LT_MAX_WORKSPACE_BYTES = 1ULL << 30;
constexpr std::uint64_t MAX_MATRIX_BYTES = 8ULL << 30;

constexpr std::array<std::int64_t, 19> CORE_M = {
    1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
    65536,262144,1048576,4194304
};
constexpr std::array<std::int64_t, 17> CORE_N = {
    1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536
};
constexpr std::array<std::int64_t, 16> N_BOUNDARY = {
    31,33,63,65,127,129,255,257,511,513,1023,1025,2047,2049,4095,4097
};
constexpr std::array<std::int64_t, 4> N_BOUNDARY_M = {
    256,4096,65536,1048576
};
constexpr std::array<std::int64_t, 18> M_BOUNDARY = {
    63,65,127,129,255,257,511,513,1023,1025,2047,2049,
    4095,4097,8191,8193,65535,65537
};
constexpr std::array<std::int64_t, 4> M_BOUNDARY_N = {
    32,128,512,4096
};
constexpr std::array<std::int64_t, 8> PRACTICAL_N = {
    96,192,384,768,1536,3072,6144,12288
};
constexpr std::array<std::int64_t, 3> PRACTICAL_M = {
    4096,65536,1048576
};

constexpr const char* RESULT_CSV = "gemv_results.csv";
constexpr const char* INTERNAL_PROFILE_ENV = "GEMV_INTERNAL_PROFILE";

const std::array<std::string, 3> IMPLEMENTATIONS = {
    "native", "rocblas_sgemv", "hipblaslt_gemm_as_gemv"
};

const std::array<std::array<int,3>,3> ORDER = {{
    {{0,1,2}}, {{1,2,0}}, {{2,0,1}}
}};

#define HIP_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) throw std::runtime_error(std::string("HIP: ") + hipGetErrorString(_e)); \
} while (0)

struct DeviceBuffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : bytes(n) {
        if (bytes) HIP_CHECK(hipMalloc(&ptr, bytes));
    }
    ~DeviceBuffer() { if (ptr) (void)hipFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) {
        o.ptr = nullptr; o.bytes = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr) (void)hipFree(ptr);
            ptr = o.ptr; bytes = o.bytes;
            o.ptr = nullptr; o.bytes = 0;
        }
        return *this;
    }
    float* f32() { return static_cast<float*>(ptr); }
};

struct HipEvent {
    hipEvent_t e = nullptr;
    HipEvent() { HIP_CHECK(hipEventCreate(&e)); }
    ~HipEvent() { if (e) (void)hipEventDestroy(e); }
};

struct Stats {
    double min_us = NAN, median_us = NAN, mean_us = NAN;
    double p95_us = NAN, p99_us = NAN, stddev_us = NAN;
};

struct Correctness {
    bool pass = false;
    double max_abs = NAN, mean_abs = NAN, max_rel = NAN;
    double rmse = NAN, relative_l2 = NAN;
    std::uint64_t failures = 0;
};

struct Profile {
    bool attempted = false;
    bool success = false;
    std::string kernel_names;
    std::uint64_t kernel_count = 0;
    double grid_size=NAN, workgroup_size=NAN, vgpr=NAN, accum_vgpr=NAN;
    double sgpr=NAN, lds_bytes=NAN, scratch_bytes=NAN;
    double grbm_count=NAN, grbm_gui_active=NAN, sq_waves=NAN;
    double fetch_kib=NAN, write_kib=NAN, gl2_hit=NAN, gl2_miss=NAN;
};

struct ResultRow {
    std::int64_t m=0,n=0,lda=0;
    std::uint64_t matrix_bytes=0;
    std::string implementation;
    std::string native_variant="NA";
    std::string status="NA";
    std::string correctness_status="NA";
    std::uint32_t iterations_per_sample=0;
    Stats timing;
    Correctness correctness;
    int lt_returned=0, lt_valid=0;
    int lt_solution_index=-1;
    std::uint64_t lt_workspace=0;
    double lt_tuning_median_us=NAN;
    Profile profile;
};

struct CaseData {
    std::int64_t m,n,lda;
    std::uint64_t matrix_bytes;
};

__host__ __device__ std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__host__ __device__ float deterministic_value(std::uint64_t index, std::uint64_t seed) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(splitmix64(index ^ seed) >> 40);
    const float unit = static_cast<float>(bits) * (1.0f / 16777216.0f);
    return unit * 2.0f - 1.0f;
}

__global__ void fill_matrix_kernel(float* A, std::uint64_t elements, std::uint64_t seed) {
    const std::uint64_t first =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i = first; i < elements; i += stride)
        A[i] = deterministic_value(i, seed);
}

__global__ void preheat_kernel(std::uint64_t rounds, float* sink) {
    float x = static_cast<float>(blockIdx.x * blockDim.x + threadIdx.x + 1);
    for (std::uint64_t i=0; i<rounds; ++i)
        x = fmaf(x, 1.000000119f, 0.000000119f);
    if ((threadIdx.x & 31) == 0) atomicAdd(sink, x);
}

void gpu_preheat(hipStream_t stream) {
    DeviceBuffer sink(sizeof(float));
    HIP_CHECK(hipMemsetAsync(sink.ptr, 0, sizeof(float), stream));
    const auto until = std::chrono::steady_clock::now() +
                       std::chrono::seconds(GPU_PREHEAT_SECONDS);
    while (std::chrono::steady_clock::now() < until)
        hipLaunchKernelGGL(preheat_kernel, dim3(256), dim3(256), 0, stream,
                           static_cast<std::uint64_t>(4096), sink.f32());
    HIP_CHECK(hipStreamSynchronize(stream));
}

std::uint32_t iterations_per_sample(std::uint64_t b) {
    if (b <= (1ULL<<20)) return 100;
    if (b <= (16ULL<<20)) return 50;
    if (b <= (256ULL<<20)) return 10;
    if (b <= (1ULL<<30)) return 3;
    return 1;
}

std::uint32_t lt_tune_iterations_per_sample(std::uint64_t b) {
    if (b <= (1ULL<<20)) return 20;
    if (b <= (16ULL<<20)) return 10;
    if (b <= (256ULL<<20)) return 3;
    return 1;
}

Stats summarize(std::vector<double> v) {
    Stats s;
    std::sort(v.begin(), v.end());
    const std::size_t n=v.size();
    s.min_us=v.front();
    s.median_us=(n%2)?v[n/2]:0.5*(v[n/2-1]+v[n/2]);
    s.mean_us=std::accumulate(v.begin(),v.end(),0.0)/static_cast<double>(n);
    auto q=[&](double p) {
        std::size_t i=static_cast<std::size_t>(std::ceil(p*n));
        i=std::max<std::size_t>(1,std::min<std::size_t>(i,n));
        return v[i-1];
    };
    s.p95_us=q(0.95); s.p99_us=q(0.99);
    double var=0;
    for(double x:v){ double d=x-s.mean_us; var+=d*d; }
    s.stddev_us=std::sqrt(var/static_cast<double>(n));
    return s;
}

std::vector<std::pair<std::int64_t,std::int64_t>> build_shapes() {
    if(std::getenv("GEMV_SMOKE")) {
        static const std::array<std::pair<std::int64_t,std::int64_t>,12> smoke = {{
            {256,4095},
            {256,6144},
            {32,32768},
            {64,65536},
            {4096,6144},
            {4096,8192},
            {16384,4096},
            {65536,4095},
            {65536,4096},
            {65536,4097},
            {4096,65536},
            {16384,65536}
        }};
        return {smoke.begin(), smoke.end()};
    }

    std::set<std::pair<std::int64_t,std::int64_t>> s;
    for(auto m:CORE_M) for(auto n:CORE_N) s.emplace(m,n);
    for(auto m:N_BOUNDARY_M) for(auto n:N_BOUNDARY) s.emplace(m,n);
    for(auto m:M_BOUNDARY) for(auto n:M_BOUNDARY_N) s.emplace(m,n);
    for(auto m:PRACTICAL_M) for(auto n:PRACTICAL_N) s.emplace(m,n);
    return {s.begin(),s.end()};
}

bool safe_matrix_bytes(std::int64_t m,std::int64_t n,std::uint64_t& bytes) {
    if(m<=0||n<=0) return false;
    const unsigned __int128 v=
        static_cast<unsigned __int128>(m)*static_cast<unsigned __int128>(n)*sizeof(float);
    if(v>std::numeric_limits<std::uint64_t>::max()) return false;
    bytes=static_cast<std::uint64_t>(v);
    return true;
}

void generate_inputs(float* A,std::int64_t m,std::int64_t n,hipStream_t stream,
                     std::vector<float>& hx,DeviceBuffer& dx) {
    const std::uint64_t elems=static_cast<std::uint64_t>(m)*static_cast<std::uint64_t>(n);
    const std::uint64_t needed=(elems+255)/256;
    const unsigned blocks=static_cast<unsigned>(
        std::max<std::uint64_t>(1,std::min<std::uint64_t>(needed,65535)));
    hipLaunchKernelGGL(fill_matrix_kernel,dim3(blocks),dim3(256),0,stream,
                       A,elems,RANDOM_SEED);
    HIP_CHECK(hipGetLastError());

    hx.resize(static_cast<std::size_t>(n));
    constexpr std::uint64_t X_SEED_XOR=0xD1B54A32D192ED03ull;
    for(std::int64_t col=0;col<n;++col)
        hx[static_cast<std::size_t>(col)] =
            deterministic_value(static_cast<std::uint64_t>(col),
                                RANDOM_SEED^X_SEED_XOR);
    HIP_CHECK(hipMemcpyAsync(dx.ptr,hx.data(),static_cast<std::size_t>(n)*sizeof(float),
                             hipMemcpyHostToDevice,stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

std::vector<double> cpu_reference(std::int64_t m,std::int64_t n,
                                  const std::vector<float>& x) {
    std::vector<double> y(static_cast<std::size_t>(m));
    for(std::int64_t row=0;row<m;++row) {
        double sum=0;
        for(std::int64_t col=0;col<n;++col) {
            const std::uint64_t idx=
                static_cast<std::uint64_t>(row)*static_cast<std::uint64_t>(n)+
                static_cast<std::uint64_t>(col);
            sum += static_cast<double>(deterministic_value(idx,RANDOM_SEED))*
                   static_cast<double>(x[static_cast<std::size_t>(col)]);
        }
        y[static_cast<std::size_t>(row)]=sum;
    }
    return y;
}

Correctness compare_to_reference(const std::vector<float>& got,
                                 const std::vector<double>& ref) {
    Correctness c;
    long double sum_abs=0,sum_sq=0,ref_sq=0;
    c.max_abs=0;c.max_rel=0;
    for(std::size_t i=0;i<got.size();++i) {
        const double e=static_cast<double>(got[i])-ref[i];
        const double ae=std::abs(e);
        const double re=ae/std::max(std::abs(ref[i]),RELATIVE_EPSILON);
        c.max_abs=std::max(c.max_abs,ae);
        c.max_rel=std::max(c.max_rel,re);
        sum_abs+=ae; sum_sq+=static_cast<long double>(e)*e;
        ref_sq+=static_cast<long double>(ref[i])*ref[i];
        if(ae>ATOL+RTOL*std::abs(ref[i])) ++c.failures;
    }
    const long double count=static_cast<long double>(got.size());
    c.mean_abs=static_cast<double>(sum_abs/count);
    c.rmse=std::sqrt(static_cast<double>(sum_sq/count));
    c.relative_l2=static_cast<double>(
        std::sqrt(sum_sq)/std::max(std::sqrt(ref_sq),
                                   static_cast<long double>(RELATIVE_EPSILON)));
    c.pass=c.failures==0;
    return c;
}

enum class Impl { Native=0,Rocblas=1,Lt=2 };

const char* impl_name(Impl i) {
    return IMPLEMENTATIONS[static_cast<int>(i)].c_str();
}

struct Runtime {
    hipStream_t stream=nullptr;
    void* rocblas=nullptr;
    void* lt=nullptr;
    float* A=nullptr;
    float* x=nullptr;
    std::array<float*,3> y{};
    void* native_workspace=nullptr;
    std::int64_t m=0,n=0,lda=0;

    int launch(Impl impl) {
        if(impl==Impl::Native)
            return static_cast<int>(
                gemv(A,x,y[0],m,n,lda,native_workspace,stream));
        if(impl==Impl::Rocblas)
            return amd_rocblas_sgemv_launch(rocblas,A,x,y[1],m,n,lda);
        return amd_lt_launch(lt,A,x,y[2],stream);
    }
    bool success(Impl impl,int code) const {
        if(impl==Impl::Native) return code==static_cast<int>(hipSuccess);
        if(impl==Impl::Rocblas) return code==amd_rocblas_success_code();
        return code==amd_lt_success_code();
    }
};

void warmup(Runtime& rt,Impl impl,int launches=WARMUP_LAUNCHES) {
    for(int i=0;i<launches;++i) {
        int code=rt.launch(impl);
        if(!rt.success(impl,code))
            throw std::runtime_error(std::string("warmup failed: ")+impl_name(impl));
    }
    HIP_CHECK(hipStreamSynchronize(rt.stream));
}

double time_batch(Runtime& rt,Impl impl,std::uint32_t iterations) {
    HipEvent start,stop;
    HIP_CHECK(hipEventRecord(start.e,rt.stream));
    for(std::uint32_t i=0;i<iterations;++i) {
        int code=rt.launch(impl);
        if(!rt.success(impl,code))
            throw std::runtime_error(std::string("timing failed: ")+impl_name(impl));
    }
    HIP_CHECK(hipEventRecord(stop.e,rt.stream));
    HIP_CHECK(hipEventSynchronize(stop.e));
    float ms=0;
    HIP_CHECK(hipEventElapsedTime(&ms,start.e,stop.e));
    return static_cast<double>(ms)*1000.0/static_cast<double>(iterations);
}

std::array<Stats,3> measure_all(Runtime& rt,std::uint32_t iterations) {
    std::array<std::vector<double>,3> samples;
    for(auto& v:samples) v.reserve(TIMING_SAMPLES);

    // Measure backends in isolated steady-state batches rather than switching
    // backend after every timed sample.  Backend order still rotates by round,
    // so no implementation permanently owns the "first" or "last" position.
    //
    // This is especially important for shapes that fit in Infinity Cache:
    // a millisecond-scale hipBLASLt kernel can otherwise evict/perturb the
    // cache and power state immediately before a tens-of-microseconds native
    // launch, creating a bimodal distribution unrelated to native steady-state
    // throughput.
    for(int round=0;round<TIMING_ROUNDS;++round) {
        for(int pos=0;pos<3;++pos) {
            const Impl impl=static_cast<Impl>(ORDER[round%3][pos]);

            // Re-establish this backend's own steady state after another
            // backend has run.  warmup() synchronizes the stream.
            warmup(rt,impl,WARMUP_LAUNCHES);

            auto& dst=samples[static_cast<int>(impl)];
            for(int s=0;s<TIMING_SAMPLES_PER_ROUND;++s)
                dst.push_back(time_batch(rt,impl,iterations));
        }
    }

    return {summarize(samples[0]),summarize(samples[1]),summarize(samples[2])};
}

std::array<Stats,2> measure_native_rocblas(Runtime& rt,
                                           std::uint32_t iterations) {
    std::array<std::vector<double>,2> samples;
    for(auto& v:samples) v.reserve(TIMING_SAMPLES);

    for(int round=0;round<TIMING_ROUNDS;++round) {
        const std::array<Impl,2> order =
            (round%2==0)
            ? std::array<Impl,2>{Impl::Native,Impl::Rocblas}
            : std::array<Impl,2>{Impl::Rocblas,Impl::Native};

        for(const Impl impl:order) {
            warmup(rt,impl,WARMUP_LAUNCHES);
            auto& dst=samples[static_cast<int>(impl)];
            for(int s=0;s<TIMING_SAMPLES_PER_ROUND;++s)
                dst.push_back(time_batch(rt,impl,iterations));
        }
    }

    return {summarize(samples[0]),summarize(samples[1])};
}

std::string csv_escape(const std::string& s) {
    if(s.find_first_of(",\"\n")==std::string::npos) return s;
    std::string o="\"";
    for(char c:s){ if(c=='"')o+="\"\""; else o+=c; }
    o+="\""; return o;
}

std::string number_or_na(double x) {
    if(!std::isfinite(x)) return "NA";
    std::ostringstream o; o<<std::setprecision(17)<<x; return o.str();
}

void write_header(std::ofstream& out) {
    out <<
    "M,N,lda,matrix_bytes,implementation,native_variant,status,correctness,"
    "seed,alpha,beta,iterations_per_sample,warmup_launches,timing_samples,"
    "timing_rounds,samples_per_round,"
    "median_us,mean_us,min_us,p95_us,p99_us,stddev_us,"
    "algorithmic_flops,algorithmic_bytes,arithmetic_intensity,gflops,effective_gbps,"
    "max_abs,mean_abs,max_rel,rmse,relative_l2,failure_count,"
    "lt_returned_algorithms,lt_valid_algorithms,lt_solution_index,lt_workspace_bytes,lt_tuning_median_us,"
    "profile_status,kernel_count,kernel_names,grid_size,workgroup_size,vgpr,accum_vgpr,sgpr,lds_bytes,scratch_bytes,"
    "grbm_count,grbm_gui_active,gpu_util,sq_waves,waves_per_output,"
    "fetch_size_kib,write_size_kib,measured_bytes,traffic_amplification,measured_arithmetic_intensity,"
    "gl2c_hit,gl2c_miss,gl2_hit_rate\n";
}

void write_row(std::ofstream& out,const ResultRow& r) {
    const double flops=2.0*static_cast<double>(r.m)*static_cast<double>(r.n);
    const double algo_bytes=4.0*(static_cast<double>(r.m)*static_cast<double>(r.n)+
                                 static_cast<double>(r.n)+static_cast<double>(r.m));
    const double ai=flops/algo_bytes;
    const double seconds=r.timing.median_us*1e-6;
    const double gflops=std::isfinite(seconds)&&seconds>0?flops/seconds/1e9:NAN;
    const double gbps=std::isfinite(seconds)&&seconds>0?algo_bytes/seconds/1e9:NAN;

    const double gpu_util=
        std::isfinite(r.profile.grbm_gui_active)&&std::isfinite(r.profile.grbm_count)&&
        r.profile.grbm_count!=0
        ?100.0*r.profile.grbm_gui_active/r.profile.grbm_count:NAN;
    const double waves_per_output=
        std::isfinite(r.profile.sq_waves)&&r.m>0
        ?r.profile.sq_waves/static_cast<double>(r.m):NAN;
    const double measured_bytes=
        std::isfinite(r.profile.fetch_kib)&&std::isfinite(r.profile.write_kib)
        ?(r.profile.fetch_kib+r.profile.write_kib)*1024.0:NAN;
    const double traffic_amp=
        std::isfinite(measured_bytes)&&algo_bytes>0?measured_bytes/algo_bytes:NAN;
    const double measured_ai=
        std::isfinite(measured_bytes)&&measured_bytes>0?flops/measured_bytes:NAN;
    const double gl2_rate=
        std::isfinite(r.profile.gl2_hit)&&std::isfinite(r.profile.gl2_miss)&&
        (r.profile.gl2_hit+r.profile.gl2_miss)>0
        ?r.profile.gl2_hit/(r.profile.gl2_hit+r.profile.gl2_miss):NAN;

    out<<r.m<<','<<r.n<<','<<r.lda<<','<<r.matrix_bytes<<','
       <<r.implementation<<','<<r.native_variant<<','<<r.status<<','<<r.correctness_status<<','
       <<RANDOM_SEED<<','<<ALPHA<<','<<BETA<<','
       <<r.iterations_per_sample<<','<<WARMUP_LAUNCHES<<','<<TIMING_SAMPLES<<','
       <<TIMING_ROUNDS<<','<<TIMING_SAMPLES_PER_ROUND<<','
       <<number_or_na(r.timing.median_us)<<','<<number_or_na(r.timing.mean_us)<<','
       <<number_or_na(r.timing.min_us)<<','<<number_or_na(r.timing.p95_us)<<','
       <<number_or_na(r.timing.p99_us)<<','<<number_or_na(r.timing.stddev_us)<<','
       <<number_or_na(flops)<<','<<number_or_na(algo_bytes)<<','<<number_or_na(ai)<<','
       <<number_or_na(gflops)<<','<<number_or_na(gbps)<<','
       <<number_or_na(r.correctness.max_abs)<<','<<number_or_na(r.correctness.mean_abs)<<','
       <<number_or_na(r.correctness.max_rel)<<','<<number_or_na(r.correctness.rmse)<<','
       <<number_or_na(r.correctness.relative_l2)<<','<<r.correctness.failures<<','
       <<(r.implementation=="hipblaslt_gemm_as_gemv"?std::to_string(r.lt_returned):"NA")<<','
       <<(r.implementation=="hipblaslt_gemm_as_gemv"?std::to_string(r.lt_valid):"NA")<<','
       <<(r.implementation=="hipblaslt_gemm_as_gemv"&&r.lt_solution_index>=0?std::to_string(r.lt_solution_index):"NA")<<','
       <<(r.implementation=="hipblaslt_gemm_as_gemv"?std::to_string(r.lt_workspace):"NA")<<','
       <<(r.implementation=="hipblaslt_gemm_as_gemv"?number_or_na(r.lt_tuning_median_us):"NA")<<','
       <<(r.profile.attempted?(r.profile.success?"PASS":"FAILED"):"NA")<<','
       <<(r.profile.success?std::to_string(r.profile.kernel_count):"NA")<<','
       <<(r.profile.success?csv_escape(r.profile.kernel_names):"NA")<<','
       <<number_or_na(r.profile.grid_size)<<','<<number_or_na(r.profile.workgroup_size)<<','
       <<number_or_na(r.profile.vgpr)<<','<<number_or_na(r.profile.accum_vgpr)<<','
       <<number_or_na(r.profile.sgpr)<<','<<number_or_na(r.profile.lds_bytes)<<','
       <<number_or_na(r.profile.scratch_bytes)<<','
       <<number_or_na(r.profile.grbm_count)<<','<<number_or_na(r.profile.grbm_gui_active)<<','
       <<number_or_na(gpu_util)<<','<<number_or_na(r.profile.sq_waves)<<','
       <<number_or_na(waves_per_output)<<','<<number_or_na(r.profile.fetch_kib)<<','
       <<number_or_na(r.profile.write_kib)<<','<<number_or_na(measured_bytes)<<','
       <<number_or_na(traffic_amp)<<','<<number_or_na(measured_ai)<<','
       <<number_or_na(r.profile.gl2_hit)<<','<<number_or_na(r.profile.gl2_miss)<<','
       <<number_or_na(gl2_rate)<<'\n';
    out.flush();
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> f;
    std::string cur;
    bool quoted=false;
    for(std::size_t i=0;i<line.size();++i) {
        char c=line[i];
        if(quoted) {
            if(c=='"'&&i+1<line.size()&&line[i+1]=='"'){cur+='"';++i;}
            else if(c=='"') quoted=false;
            else cur+=c;
        } else {
            if(c=='"') quoted=true;
            else if(c==','){f.push_back(cur);cur.clear();}
            else cur+=c;
        }
    }
    f.push_back(cur);
    return f;
}

double to_double_or_nan(const std::string& s) {
    try{return std::stod(s);}catch(...){return NAN;}
}

struct MarkerRange {
    std::string key;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
};

struct ProfileCase {
    std::int64_t m = 0;
    std::int64_t n = 0;
    std::array<bool,3> enabled{{false,false,false}};
    int lt_solution_index = -1;
};

std::string self_executable() {
    std::array<char,4096> b{};
    const auto n=::readlink("/proc/self/exe",b.data(),b.size()-1);
    if(n<=0)throw std::runtime_error("cannot resolve /proc/self/exe");
    b[static_cast<std::size_t>(n)]='\0';
    return std::string(b.data());
}

std::string shell_quote(const std::string& s) {
    std::string o="'";
    for(char c:s){if(c=='\'')o+="'\\''";else o+=c;}
    o+="'";return o;
}

std::string yaml_quote(const std::string& s) {
    std::string o="\"";
    for(char c:s) {
        if(c=='\\') o+="\\\\";
        else if(c=='"') o+="\\\"";
        else o+=c;
    }
    o+="\"";
    return o;
}

std::string profile_key(std::int64_t m,std::int64_t n,int impl) {
    return "GEMV|M="+std::to_string(m)+"|N="+std::to_string(n)+
           "|I="+std::to_string(impl);
}

int implementation_index(const std::string& name) {
    for(int i=0;i<3;++i)
        if(IMPLEMENTATIONS[static_cast<std::size_t>(i)]==name) return i;
    return -1;
}

std::uint64_t to_u64_or_zero(const std::string& s) {
    try{return static_cast<std::uint64_t>(std::stoull(s));}
    catch(...){return 0;}
}

std::set<std::string> available_counters() {
    const fs::path tmp=".gemv_rocprof_avail.txt";
    const std::string cmd="/opt/rocm/bin/rocprofv3 --list-avail > "+
                          shell_quote(tmp.string())+" 2>&1";
    std::set<std::string> result;
    if(std::system(cmd.c_str())!=0){fs::remove(tmp);return result;}
    std::ifstream in(tmp);
    std::string text((std::istreambuf_iterator<char>(in)),{});
    fs::remove(tmp);
    const std::array<std::string,7> wanted={
        "SQ_WAVES","GRBM_COUNT","GRBM_GUI_ACTIVE",
        "FETCH_SIZE","WRITE_SIZE","GL2C_HIT","GL2C_MISS"
    };
    for(const auto& c:wanted)if(text.find(c)!=std::string::npos)result.insert(c);
    return result;
}

void write_profile_state_from_results(const fs::path& results,const fs::path& state) {
    std::ifstream in(results);
    if(!in)throw std::runtime_error("cannot read result CSV for profiling");

    std::string header_line;
    if(!std::getline(in,header_line))
        throw std::runtime_error("empty result CSV");
    const auto header=parse_csv_line(header_line);
    std::unordered_map<std::string,std::size_t> col;
    for(std::size_t i=0;i<header.size();++i)col[header[i]]=i;

    const std::array<const char*,5> required={
        "M","N","implementation","correctness","lt_solution_index"
    };
    for(auto name:required)
        if(!col.count(name))
            throw std::runtime_error(std::string("missing CSV column: ")+name);

    std::ofstream out(state,std::ios::trunc);
    if(!out)throw std::runtime_error("cannot create profile state");
    out<<"M,N,implementation_index,lt_solution_index\n";

    std::string line;
    while(std::getline(in,line)) {
        auto f=parse_csv_line(line);
        auto get=[&](const std::string& k)->std::string{
            const auto i=col.at(k);
            return i<f.size()?f[i]:"";
        };
        if(get("correctness")!="PASS")continue;
        const int impl=implementation_index(get("implementation"));
        if(impl<0)continue;

        int solution=-1;
        if(impl==static_cast<int>(Impl::Lt)) {
            try{solution=std::stoi(get("lt_solution_index"));}
            catch(...){continue;}
        }

        out<<get("M")<<','<<get("N")<<','<<impl<<','<<solution<<'\n';
    }
}

std::vector<ProfileCase> read_profile_state(const fs::path& state) {
    std::ifstream in(state);
    if(!in)throw std::runtime_error("cannot open profile state");

    std::string line;
    std::getline(in,line);

    std::map<std::pair<std::int64_t,std::int64_t>,ProfileCase> grouped;
    while(std::getline(in,line)) {
        const auto f=parse_csv_line(line);
        if(f.size()<4)continue;
        const std::int64_t m=std::stoll(f[0]);
        const std::int64_t n=std::stoll(f[1]);
        const int impl=std::stoi(f[2]);
        const int solution=std::stoi(f[3]);
        if(impl<0||impl>=3)continue;

        auto& c=grouped[{m,n}];
        c.m=m;c.n=n;
        c.enabled[static_cast<std::size_t>(impl)]=true;
        if(impl==static_cast<int>(Impl::Lt))
            c.lt_solution_index=solution;
    }

    std::vector<ProfileCase> result;
    result.reserve(grouped.size());
    for(auto& kv:grouped)result.push_back(kv.second);
    return result;
}

bool write_profile_yaml(const fs::path& path,
                        const std::set<std::string>& avail,
                        const fs::path& base) {
    struct Job {
        std::vector<std::string> counters;
        std::string directory;
    };
    std::vector<Job> jobs;

    auto make_job=[&](std::initializer_list<const char*> names,const char* dir){
        Job j;j.directory=dir;
        for(auto n:names)if(avail.count(n))j.counters.emplace_back(n);
        if(!j.counters.empty())jobs.push_back(std::move(j));
    };

    make_job({"SQ_WAVES","GRBM_COUNT","GRBM_GUI_ACTIVE"},"pass_execution");
    make_job({"FETCH_SIZE","WRITE_SIZE"},"pass_memory");
    make_job({"GL2C_HIT","GL2C_MISS"},"pass_gl2");
    if(jobs.empty())return false;

    std::ofstream out(path,std::ios::trunc);
    if(!out)return false;
    out<<"jobs:\n";
    for(const auto& job:jobs) {
        const fs::path dir=fs::absolute(base/job.directory);
        fs::create_directories(dir);
        out<<"  - pmc: [";
        for(std::size_t i=0;i<job.counters.size();++i) {
            if(i)out<<", ";
            out<<'"'<<job.counters[i]<<'"';
        }
        out<<"]\n";
        out<<"    marker_trace: true\n";
        out<<"    output_format: [\"csv\"]\n";
        out<<"    output_directory: "<<yaml_quote(dir.string())<<"\n";
    }
    return true;
}

std::vector<MarkerRange> load_marker_ranges(const fs::path& dir) {
    std::vector<MarkerRange> ranges;
    if(!fs::exists(dir))return ranges;

    for(const auto& entry:fs::recursive_directory_iterator(dir)) {
        if(!entry.is_regular_file())continue;
        if(entry.path().filename().string().find("marker_api_trace.csv")==std::string::npos)
            continue;

        std::ifstream in(entry.path());
        std::string header_line;
        if(!std::getline(in,header_line))continue;
        const auto header=parse_csv_line(header_line);
        std::unordered_map<std::string,std::size_t> col;
        for(std::size_t i=0;i<header.size();++i)col[header[i]]=i;
        if(!col.count("Function")||!col.count("Start_Timestamp")||!col.count("End_Timestamp"))
            continue;

        std::string line;
        while(std::getline(in,line)) {
            const auto f=parse_csv_line(line);
            auto get=[&](const std::string& k)->std::string{
                const auto it=col.find(k);
                return(it!=col.end()&&it->second<f.size())?f[it->second]:"";
            };
            const std::string fn=get("Function");
            if(fn.rfind("GEMV|",0)!=0)continue;
            MarkerRange r;
            r.key=fn;
            r.start=to_u64_or_zero(get("Start_Timestamp"));
            r.end=to_u64_or_zero(get("End_Timestamp"));
            if(r.end>=r.start)ranges.push_back(std::move(r));
        }
    }

    std::sort(ranges.begin(),ranges.end(),
              [](const MarkerRange& a,const MarkerRange& b){return a.start<b.start;});
    return ranges;
}

const MarkerRange* containing_range(const std::vector<MarkerRange>& ranges,
                                    std::uint64_t start,std::uint64_t end) {
    auto it=std::upper_bound(
        ranges.begin(),ranges.end(),start,
        [](std::uint64_t value,const MarkerRange& r){return value<r.start;});
    if(it==ranges.begin())return nullptr;
    --it;
    if(start>=it->start&&end<=it->end)return &*it;
    return nullptr;
}

void add_counter(Profile& p,const std::string& name,double value) {
    if(!std::isfinite(value))return;
    double* dst=nullptr;
    if(name=="GRBM_COUNT")dst=&p.grbm_count;
    else if(name=="GRBM_GUI_ACTIVE")dst=&p.grbm_gui_active;
    else if(name=="SQ_WAVES")dst=&p.sq_waves;
    else if(name=="FETCH_SIZE")dst=&p.fetch_kib;
    else if(name=="WRITE_SIZE")dst=&p.write_kib;
    else if(name=="GL2C_HIT")dst=&p.gl2_hit;
    else if(name=="GL2C_MISS")dst=&p.gl2_miss;
    if(!dst)return;
    if(!std::isfinite(*dst))*dst=0.0;
    *dst+=value;
}

void parse_profile_pass(const fs::path& dir,
                        bool metadata_pass,
                        std::map<std::string,Profile>& profiles) {
    const auto ranges=load_marker_ranges(dir);
    if(ranges.empty())return;

    std::map<std::string,std::set<std::string>> dispatches;
    std::map<std::string,std::set<std::string>> names;

    for(const auto& entry:fs::recursive_directory_iterator(dir)) {
        if(!entry.is_regular_file())continue;
        if(entry.path().filename().string().find("counter_collection.csv")==std::string::npos)
            continue;

        std::ifstream in(entry.path());
        std::string header_line;
        if(!std::getline(in,header_line))continue;
        const auto header=parse_csv_line(header_line);
        std::unordered_map<std::string,std::size_t> col;
        for(std::size_t i=0;i<header.size();++i)col[header[i]]=i;

        const std::array<const char*,4> req={
            "Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"
        };
        bool ok=true;
        for(auto n:req)if(!col.count(n))ok=false;
        if(!ok)continue;

        std::string line;
        while(std::getline(in,line)) {
            const auto f=parse_csv_line(line);
            auto get=[&](const std::string& k)->std::string{
                const auto it=col.find(k);
                return(it!=col.end()&&it->second<f.size())?f[it->second]:"";
            };

            const std::uint64_t start=to_u64_or_zero(get("Start_Timestamp"));
            const std::uint64_t end=to_u64_or_zero(get("End_Timestamp"));
            const auto* range=containing_range(ranges,start,end);
            if(!range)continue;

            auto& p=profiles[range->key];
            p.attempted=true;
            const std::string counter=get("Counter_Name");
            add_counter(p,counter,to_double_or_nan(get("Counter_Value")));

            if(metadata_pass&&counter=="SQ_WAVES") {
                const std::string dispatch=get("Dispatch_Id");
                if(!dispatch.empty())dispatches[range->key].insert(dispatch);
                const std::string kernel=get("Kernel_Name");
                if(!kernel.empty())names[range->key].insert(kernel);

                auto mx=[&](const char* key,double& dst){
                    const double v=to_double_or_nan(get(key));
                    if(std::isfinite(v)&&(!std::isfinite(dst)||v>dst))dst=v;
                };
                mx("Grid_Size",p.grid_size);
                mx("Workgroup_Size",p.workgroup_size);
                mx("VGPR_Count",p.vgpr);
                mx("Accum_VGPR_Count",p.accum_vgpr);
                mx("SGPR_Count",p.sgpr);
                mx("LDS_Block_Size",p.lds_bytes);
                mx("Scratch_Size",p.scratch_bytes);
            }
        }
    }

    if(metadata_pass) {
        for(auto& kv:dispatches) {
            auto& p=profiles[kv.first];
            p.kernel_count=kv.second.size();
            std::ostringstream out;
            bool first=true;
            for(const auto& name:names[kv.first]) {
                if(!first)out<<" | ";
                out<<name;first=false;
            }
            p.kernel_names=out.str();
        }
    }
}

bool any_profile_counter(const Profile& p) {
    return std::isfinite(p.grbm_count)||std::isfinite(p.grbm_gui_active)||
           std::isfinite(p.sq_waves)||std::isfinite(p.fetch_kib)||
           std::isfinite(p.write_kib)||std::isfinite(p.gl2_hit)||
           std::isfinite(p.gl2_miss);
}

bool complete_profile(const Profile& p) {
    return std::isfinite(p.grbm_count)&&std::isfinite(p.grbm_gui_active)&&
           std::isfinite(p.sq_waves)&&std::isfinite(p.fetch_kib)&&
           std::isfinite(p.write_kib)&&std::isfinite(p.gl2_hit)&&
           std::isfinite(p.gl2_miss);
}

std::map<std::string,Profile> run_global_profiles(
    const fs::path& state,const std::set<std::string>& avail) {
    const fs::path base=".gemv_profile_tmp";
    fs::remove_all(base);
    fs::create_directories(base);

    std::map<std::string,Profile> profiles;
    for(const auto& c:read_profile_state(state)) {
        for(int i=0;i<3;++i) {
            if(c.enabled[static_cast<std::size_t>(i)]) {
                auto& p=profiles[profile_key(c.m,c.n,i)];
                p.attempted=true;
            }
        }
    }

    const fs::path yaml=base/"profile.yaml";
    if(!write_profile_yaml(yaml,avail,base))
        return profiles;

    const fs::path log=base/"rocprofv3.log";
    std::ostringstream cmd;
    cmd<<"GEMV_INTERNAL_PROFILE=1 "
       <<"GEMV_PROFILE_STATE="<<shell_quote(fs::absolute(state).string())<<' '
       <<"/opt/rocm/bin/rocprofv3 -i "<<shell_quote(fs::absolute(yaml).string())
       <<" -- "<<shell_quote(self_executable())
       <<" > "<<shell_quote(log.string())<<" 2>&1";

    std::cout<<"Global PMC profiling: 3 passes over all correctness-PASS cases\n";
    const int rc=std::system(cmd.str().c_str());
    if(rc!=0)
        std::cerr<<"rocprofv3 global profiling returned "<<rc
                 <<"; see "<<log.string()<<"\n";

    parse_profile_pass(base/"pass_execution",true,profiles);
    parse_profile_pass(base/"pass_memory",false,profiles);
    parse_profile_pass(base/"pass_gl2",false,profiles);

    for(auto& kv:profiles)
        kv.second.success=any_profile_counter(kv.second);

    return profiles;
}

int internal_profile_main() {
    (void)roctxProfilerPause(0);

    const char* state_env=std::getenv("GEMV_PROFILE_STATE");
    if(!state_env)return 3;
    const auto cases=read_profile_state(state_env);

    HIP_CHECK(hipSetDevice(0));
    hipStream_t stream=nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));

    for(const auto& c:cases) {
        std::uint64_t matrix_bytes=0;
        if(!safe_matrix_bytes(c.m,c.n,matrix_bytes)||
           matrix_bytes>MAX_MATRIX_BYTES)continue;

        try {
            DeviceBuffer A(static_cast<std::size_t>(matrix_bytes));
            DeviceBuffer x(static_cast<std::size_t>(c.n)*sizeof(float));
            DeviceBuffer y(static_cast<std::size_t>(c.m)*sizeof(float));
            std::vector<float> hx;
            generate_inputs(A.f32(),c.m,c.n,stream,hx,x);

            void* rb=nullptr;
            void* lt=nullptr;

            if(c.enabled[static_cast<int>(Impl::Rocblas)]) {
                rb=amd_rocblas_create(stream);
                if(!rb)throw std::runtime_error("profile rocBLAS create failed");
            }

            if(c.enabled[static_cast<int>(Impl::Lt)]) {
                lt=amd_lt_create(c.m,c.n,c.n,LT_MAX_WORKSPACE_BYTES);
                if(!lt)throw std::runtime_error("profile hipBLASLt create failed");
                if(!amd_lt_load_solution_index(lt,c.lt_solution_index))
                    throw std::runtime_error("profile hipBLASLt solution index restore failed");
            }

            DeviceBuffer native_workspace(
                static_cast<std::size_t>(gemv_workspace_bytes(c.m,c.n)));
            if(native_workspace.bytes) {
                HIP_CHECK(hipMemsetAsync(
                    native_workspace.ptr,0,native_workspace.bytes,stream));
                HIP_CHECK(hipStreamSynchronize(stream));
            }

            Runtime rt;
            rt.stream=stream;rt.rocblas=rb;rt.lt=lt;
            rt.A=A.f32();rt.x=x.f32();rt.m=c.m;rt.n=c.n;rt.lda=c.n;
            rt.y={y.f32(),y.f32(),y.f32()};
            rt.native_workspace=native_workspace.ptr;

            for(int i=0;i<3;++i) {
                if(!c.enabled[static_cast<std::size_t>(i)])continue;
                const Impl impl=static_cast<Impl>(i);
                warmup(rt,impl,PROFILE_WARMUP_LAUNCHES);

                const std::string label=profile_key(c.m,c.n,i);
                (void)roctxProfilerResume(0);
                (void)roctxRangePush(label.c_str());

                const int code=rt.launch(impl);
                if(!rt.success(impl,code)) {
                    (void)roctxRangePop();
                    (void)roctxProfilerPause(0);
                    throw std::runtime_error("profile target launch failed");
                }

                HIP_CHECK(hipStreamSynchronize(stream));
                (void)roctxRangePop();
                (void)roctxProfilerPause(0);
            }

            if(lt)amd_lt_destroy(lt);
            if(rb)amd_rocblas_destroy(rb);
        }
        catch(const std::exception& e) {
            (void)roctxProfilerPause(0);
            std::cerr<<"profile case M="<<c.m<<" N="<<c.n<<": "<<e.what()<<"\n";
        }
    }

    (void)hipStreamDestroy(stream);
    return 0;
}

void merge_profiles_into_csv(const fs::path& csv_path,
                             const std::map<std::string,Profile>& profiles) {
    std::ifstream in(csv_path);
    if(!in)throw std::runtime_error("cannot reopen result CSV");

    std::string header_line;
    if(!std::getline(in,header_line))throw std::runtime_error("empty result CSV");
    const auto header=parse_csv_line(header_line);
    std::unordered_map<std::string,std::size_t> col;
    for(std::size_t i=0;i<header.size();++i)col[header[i]]=i;

    const fs::path tmp=csv_path.string()+".tmp";
    std::ofstream out(tmp,std::ios::trunc);
    if(!out)throw std::runtime_error("cannot create merged CSV");
    out<<header_line<<'\n';

    auto set_field=[&](std::vector<std::string>& f,const std::string& name,
                       const std::string& value) {
        const auto it=col.find(name);
        if(it!=col.end()&&it->second<f.size())f[it->second]=value;
    };

    std::string line;
    while(std::getline(in,line)) {
        auto f=parse_csv_line(line);
        auto get=[&](const std::string& name)->std::string{
            const auto it=col.find(name);
            return(it!=col.end()&&it->second<f.size())?f[it->second]:"";
        };

        if(get("correctness")=="PASS") {
            const std::int64_t m=std::stoll(get("M"));
            const std::int64_t n=std::stoll(get("N"));
            const int impl=implementation_index(get("implementation"));
            const auto it=profiles.find(profile_key(m,n,impl));

            if(it!=profiles.end()) {
                const Profile& p=it->second;
                const bool complete=complete_profile(p);
                const bool any=any_profile_counter(p);
                set_field(f,"profile_status",complete?"PASS":(any?"PARTIAL":"FAILED"));

                if(p.kernel_count>0)set_field(f,"kernel_count",std::to_string(p.kernel_count));
                if(!p.kernel_names.empty())set_field(f,"kernel_names",p.kernel_names);
                set_field(f,"grid_size",number_or_na(p.grid_size));
                set_field(f,"workgroup_size",number_or_na(p.workgroup_size));
                set_field(f,"vgpr",number_or_na(p.vgpr));
                set_field(f,"accum_vgpr",number_or_na(p.accum_vgpr));
                set_field(f,"sgpr",number_or_na(p.sgpr));
                set_field(f,"lds_bytes",number_or_na(p.lds_bytes));
                set_field(f,"scratch_bytes",number_or_na(p.scratch_bytes));
                set_field(f,"grbm_count",number_or_na(p.grbm_count));
                set_field(f,"grbm_gui_active",number_or_na(p.grbm_gui_active));
                set_field(f,"sq_waves",number_or_na(p.sq_waves));
                set_field(f,"fetch_size_kib",number_or_na(p.fetch_kib));
                set_field(f,"write_size_kib",number_or_na(p.write_kib));
                set_field(f,"gl2c_hit",number_or_na(p.gl2_hit));
                set_field(f,"gl2c_miss",number_or_na(p.gl2_miss));

                const double dm=static_cast<double>(m),dn=static_cast<double>(n);
                const double flops=2.0*dm*dn;
                const double algorithmic_bytes=4.0*(dm*dn+dn+dm);
                const double gpu_util=
                    std::isfinite(p.grbm_gui_active)&&std::isfinite(p.grbm_count)&&
                    p.grbm_count!=0?100.0*p.grbm_gui_active/p.grbm_count:NAN;
                const double waves_per_output=
                    std::isfinite(p.sq_waves)&&m>0?p.sq_waves/dm:NAN;
                const double measured_bytes=
                    std::isfinite(p.fetch_kib)&&std::isfinite(p.write_kib)
                    ?(p.fetch_kib+p.write_kib)*1024.0:NAN;
                const double traffic_amp=
                    std::isfinite(measured_bytes)&&algorithmic_bytes>0
                    ?measured_bytes/algorithmic_bytes:NAN;
                const double measured_ai=
                    std::isfinite(measured_bytes)&&measured_bytes>0
                    ?flops/measured_bytes:NAN;
                const double gl2_rate=
                    std::isfinite(p.gl2_hit)&&std::isfinite(p.gl2_miss)&&
                    (p.gl2_hit+p.gl2_miss)>0
                    ?p.gl2_hit/(p.gl2_hit+p.gl2_miss):NAN;

                set_field(f,"gpu_util",number_or_na(gpu_util));
                set_field(f,"waves_per_output",number_or_na(waves_per_output));
                set_field(f,"measured_bytes",number_or_na(measured_bytes));
                set_field(f,"traffic_amplification",number_or_na(traffic_amp));
                set_field(f,"measured_arithmetic_intensity",number_or_na(measured_ai));
                set_field(f,"gl2_hit_rate",number_or_na(gl2_rate));
            } else {
                set_field(f,"profile_status","FAILED");
            }
        }

        for(std::size_t i=0;i<f.size();++i) {
            if(i)out<<',';
            out<<csv_escape(f[i]);
        }
        out<<'\n';
    }

    out.close();
    in.close();
    fs::rename(tmp,csv_path);
}

void write_skipped_rows(std::ofstream& out,std::int64_t m,std::int64_t n,
                        std::uint64_t bytes,const std::string& status) {
    for(const auto& impl:IMPLEMENTATIONS) {
        ResultRow r;
        r.m=m;r.n=n;r.lda=n;r.matrix_bytes=bytes;
        r.implementation=impl;r.status=status;r.correctness_status="NA";
        r.iterations_per_sample=iterations_per_sample(bytes);
        write_row(out,r);
    }
}

}

int main() {
    try {
        if(std::getenv(INTERNAL_PROFILE_ENV))
            return internal_profile_main();

        HIP_CHECK(hipSetDevice(0));
        hipDeviceProp_t props{};
        HIP_CHECK(hipGetDeviceProperties(&props,0));
        std::cout<<"GPU="<<props.name<<" arch="<<props.gcnArchName<<"\n";
        std::cout<<"ROCm contract=7.2.4\n";

        hipStream_t stream=nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));

        const auto avail=available_counters();
        std::cout<<"rocprofv3 counters:";
        for(const auto& c:avail)std::cout<<' '<<c;
        std::cout<<"\n";

        std::ofstream csv(RESULT_CSV,std::ios::trunc);
        if(!csv)throw std::runtime_error("cannot create gemv_results.csv");
        write_header(csv);

        gpu_preheat(stream);

        const auto shapes=build_shapes();
        std::size_t index=0;

        for(const auto& shape:shapes) {
            const std::int64_t m=shape.first,n=shape.second;
            ++index;
            std::uint64_t matrix_bytes=0;

            if(!safe_matrix_bytes(m,n,matrix_bytes)) {
                write_skipped_rows(csv,m,n,0,"INVALID_SIZE");
                continue;
            }
            if(matrix_bytes>MAX_MATRIX_BYTES) {
                write_skipped_rows(csv,m,n,matrix_bytes,"SKIPPED_MATRIX_SIZE");
                continue;
            }

            std::cout<<"["<<index<<"/"<<shapes.size()<<"] M="<<m<<" N="<<n
                     <<" A="<<std::fixed<<std::setprecision(3)
                     <<static_cast<double>(matrix_bytes)/(1024.0*1024.0)<<" MiB\n";

            std::array<ResultRow,3> rows;
            for(int i=0;i<3;++i) {
                rows[i].m=m;rows[i].n=n;rows[i].lda=n;
                rows[i].matrix_bytes=matrix_bytes;
                rows[i].implementation=IMPLEMENTATIONS[i];
                rows[i].native_variant=(i==0)?gemv_path_name(m,n):"NA";
                rows[i].iterations_per_sample=iterations_per_sample(matrix_bytes);
            }

            try {
                DeviceBuffer A(static_cast<std::size_t>(matrix_bytes));
                DeviceBuffer x(static_cast<std::size_t>(n)*sizeof(float));
                std::array<DeviceBuffer,3> y={
                    DeviceBuffer(static_cast<std::size_t>(m)*sizeof(float)),
                    DeviceBuffer(static_cast<std::size_t>(m)*sizeof(float)),
                    DeviceBuffer(static_cast<std::size_t>(m)*sizeof(float))
                };

                std::vector<float> hx;
                generate_inputs(A.f32(),m,n,stream,hx,x);
                const auto reference=cpu_reference(m,n,hx);

                void* rb=amd_rocblas_create(stream);
                void* lt=amd_lt_create(m,n,n,LT_MAX_WORKSPACE_BYTES);
                if(!rb)throw std::runtime_error("rocBLAS handle creation failed");
                if(!lt){amd_rocblas_destroy(rb);throw std::runtime_error("hipBLASLt creation failed");}

                int lt_returned=0,lt_valid=0,lt_solution_index=-1;
                std::uint64_t lt_workspace=0;
                double lt_tune_us=NAN;
                const int tune_code=amd_lt_tune(
                    lt,A.f32(),x.f32(),y[2].f32(),stream,
                    LT_HEURISTIC_MAX_ALGOS,LT_TUNE_WARMUP,LT_TUNE_SAMPLES,
                    static_cast<int>(lt_tune_iterations_per_sample(matrix_bytes)),
                    &lt_returned,&lt_valid,&lt_solution_index,&lt_workspace,&lt_tune_us);
                const bool lt_supported=tune_code==amd_lt_success_code();

                DeviceBuffer native_workspace(
                    static_cast<std::size_t>(gemv_workspace_bytes(m,n)));
                if(native_workspace.bytes) {
                    HIP_CHECK(hipMemsetAsync(
                        native_workspace.ptr,0,native_workspace.bytes,stream));
                    HIP_CHECK(hipStreamSynchronize(stream));
                }

                Runtime rt;
                rt.stream=stream;rt.rocblas=rb;rt.lt=lt;
                rt.A=A.f32();rt.x=x.f32();rt.m=m;rt.n=n;rt.lda=n;
                rt.y={y[0].f32(),y[1].f32(),y[2].f32()};
                rt.native_workspace=native_workspace.ptr;

                if(lt_supported) {
                    const auto stats=measure_all(rt,rows[0].iterations_per_sample);
                    for(int i=0;i<3;++i)rows[i].timing=stats[i];
                } else {
                    const auto stats=
                        measure_native_rocblas(rt,rows[0].iterations_per_sample);
                    rows[0].timing=stats[0];
                    rows[1].timing=stats[1];
                    rows[2].status="UNSUPPORTED";
                }

                std::array<std::vector<float>,3> host_y;
                for(int i=0;i<3;++i) {
                    if(i==2&&!lt_supported)continue;
                    host_y[i].resize(static_cast<std::size_t>(m));
                    HIP_CHECK(hipMemcpyAsync(host_y[i].data(),y[i].ptr,
                                             static_cast<std::size_t>(m)*sizeof(float),
                                             hipMemcpyDeviceToHost,stream));
                }
                HIP_CHECK(hipStreamSynchronize(stream));

                for(int i=0;i<3;++i) {
                    if(i==2&&!lt_supported) {
                        rows[i].correctness_status="NA";
                        continue;
                    }
                    rows[i].correctness=compare_to_reference(host_y[i],reference);
                    rows[i].correctness_status=rows[i].correctness.pass?"PASS":"FAIL";
                    rows[i].status=rows[i].correctness.pass?"PASS":"CORRECTNESS_FAIL";
                }

                rows[2].lt_returned=lt_returned;
                rows[2].lt_valid=lt_valid;
                rows[2].lt_solution_index=lt_solution_index;
                rows[2].lt_workspace=lt_workspace;
                rows[2].lt_tuning_median_us=lt_tune_us;

                amd_lt_destroy(lt);
                amd_rocblas_destroy(rb);

                for(const auto& r:rows)write_row(csv,r);
            }
            catch(const std::bad_alloc&) {
                write_skipped_rows(csv,m,n,matrix_bytes,"ALLOCATION_FAILED");
            }
            catch(const std::exception& e) {
                std::cerr<<"case error M="<<m<<" N="<<n<<": "<<e.what()<<"\n";
                for(auto& r:rows) {
                    if(r.status=="NA")r.status="CASE_ERROR";
                    write_row(csv,r);
                }
            }
        }

        csv.close();
        (void)hipStreamDestroy(stream);

        const fs::path profile_state=".gemv_profile_state.csv";
        if(!std::getenv("GEMV_SKIP_PROFILING")) {
            write_profile_state_from_results(RESULT_CSV,profile_state);
            const auto profiles=run_global_profiles(profile_state,avail);
            merge_profiles_into_csv(RESULT_CSV,profiles);
            fs::remove(profile_state);
            fs::remove_all(".gemv_profile_tmp");
        } else {
            std::cout<<"Skipping rocprofv3 because GEMV_SKIP_PROFILING is set\n";
        }

        std::cout<<"Wrote "<<RESULT_CSV<<"\n";
        return 0;
    }
    catch(const std::exception& e) {
        std::cerr<<"fatal: "<<e.what()<<"\n";
        return 1;
    }
}

namespace {

struct RocblasContext {
    rocblas_handle handle = nullptr;
};

struct LtContext {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;
    hipblasLtMatrixLayout_t d = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;

    hipblasLtMatmulAlgo_t selected_algo{};
    bool has_selected_algo = false;
    int selected_solution_index = -1;
    std::size_t selected_workspace_bytes = 0;
    void* workspace = nullptr;
    std::uint64_t max_workspace_bytes = 0;
};

void destroy_lt(LtContext* ctx) {
    if (!ctx) return;
    if (ctx->workspace) (void)hipFree(ctx->workspace);
    if (ctx->pref) hipblasLtMatmulPreferenceDestroy(ctx->pref);
    if (ctx->d) hipblasLtMatrixLayoutDestroy(ctx->d);
    if (ctx->c) hipblasLtMatrixLayoutDestroy(ctx->c);
    if (ctx->b) hipblasLtMatrixLayoutDestroy(ctx->b);
    if (ctx->a) hipblasLtMatrixLayoutDestroy(ctx->a);
    if (ctx->op) hipblasLtMatmulDescDestroy(ctx->op);
    if (ctx->handle) hipblasLtDestroy(ctx->handle);
    delete ctx;
}

hipblasStatus_t lt_launch_algo(LtContext* ctx,
                               const hipblasLtMatmulAlgo_t* algo,
                               std::size_t workspace_bytes,
                               const float* A,
                               const float* x,
                               float* y,
                               hipStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return hipblasLtMatmul(ctx->handle, ctx->op,
                           &alpha,
                           A, ctx->a,
                           x, ctx->b,
                           &beta,
                           y, ctx->c,
                           y, ctx->d,
                           algo,
                           workspace_bytes ? ctx->workspace : nullptr,
                           workspace_bytes,
                           stream);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

}

extern "C" void* amd_rocblas_create(hipStream_t stream) {
    auto* ctx = new(std::nothrow) RocblasContext;
    if (!ctx) return nullptr;

    if (rocblas_create_handle(&ctx->handle) != rocblas_status_success) {
        delete ctx;
        return nullptr;
    }

    if (rocblas_set_stream(ctx->handle, stream) != rocblas_status_success ||
        rocblas_set_pointer_mode(ctx->handle, rocblas_pointer_mode_host) != rocblas_status_success) {
        rocblas_destroy_handle(ctx->handle);
        delete ctx;
        return nullptr;
    }
    return ctx;
}

extern "C" void amd_rocblas_destroy(void* opaque) {
    auto* ctx = static_cast<RocblasContext*>(opaque);
    if (!ctx) return;
    if (ctx->handle) rocblas_destroy_handle(ctx->handle);
    delete ctx;
}

extern "C" int amd_rocblas_sgemv_launch(void* opaque,
                                        const float* A,
                                        const float* x,
                                        float* y,
                                        std::int64_t m,
                                        std::int64_t n,
                                        std::int64_t lda) {
    auto* ctx = static_cast<RocblasContext*>(opaque);
    if (!ctx || !ctx->handle || !A || !x || !y)
        return static_cast<int>(rocblas_status_invalid_handle);

    if (m <= 0 || n <= 0 || lda < n ||
        m > std::numeric_limits<rocblas_int>::max() ||
        n > std::numeric_limits<rocblas_int>::max() ||
        lda > std::numeric_limits<rocblas_int>::max()) {
        return static_cast<int>(rocblas_status_invalid_size);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    return static_cast<int>(
        rocblas_sgemv(static_cast<RocblasContext*>(opaque)->handle,
                      rocblas_operation_transpose,
                      static_cast<rocblas_int>(n),
                      static_cast<rocblas_int>(m),
                      &alpha,
                      A,
                      static_cast<rocblas_int>(lda),
                      x,
                      1,
                      &beta,
                      y,
                      1));
}

extern "C" int amd_rocblas_success_code() {
    return static_cast<int>(rocblas_status_success);
}

extern "C" void* amd_lt_create(std::int64_t m,
                               std::int64_t n,
                               std::int64_t lda,
                               std::uint64_t max_workspace_bytes) {
    if (m <= 0 || n <= 0 || lda < n) return nullptr;

    auto* ctx = new(std::nothrow) LtContext;
    if (!ctx) return nullptr;
    ctx->max_workspace_bytes = max_workspace_bytes;

    if (hipblasLtCreate(&ctx->handle) != HIPBLAS_STATUS_SUCCESS) {
        destroy_lt(ctx); return nullptr;
    }

    if (hipblasLtMatmulDescCreate(&ctx->op, HIPBLAS_COMPUTE_32F, HIP_R_32F)
        != HIPBLAS_STATUS_SUCCESS) {
        destroy_lt(ctx); return nullptr;
    }

    hipblasOperation_t trans_a = HIPBLAS_OP_T;
    hipblasOperation_t trans_b = HIPBLAS_OP_N;

    if (hipblasLtMatmulDescSetAttribute(
            ctx->op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a))
            != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(
            ctx->op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b))
            != HIPBLAS_STATUS_SUCCESS) {
        destroy_lt(ctx); return nullptr;
    }

    if (hipblasLtMatrixLayoutCreate(
            &ctx->a, HIP_R_32F, static_cast<std::uint64_t>(n),
            static_cast<std::uint64_t>(m), lda) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(
            &ctx->b, HIP_R_32F, static_cast<std::uint64_t>(n), 1, n)
            != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(
            &ctx->c, HIP_R_32F, static_cast<std::uint64_t>(m), 1, m)
            != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(
            &ctx->d, HIP_R_32F, static_cast<std::uint64_t>(m), 1, m)
            != HIPBLAS_STATUS_SUCCESS) {
        destroy_lt(ctx); return nullptr;
    }

    if (hipblasLtMatmulPreferenceCreate(&ctx->pref) != HIPBLAS_STATUS_SUCCESS) {
        destroy_lt(ctx); return nullptr;
    }

    const std::uint64_t limit = max_workspace_bytes;
    if (hipblasLtMatmulPreferenceSetAttribute(
            ctx->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &limit, sizeof(limit)) != HIPBLAS_STATUS_SUCCESS) {
        destroy_lt(ctx); return nullptr;
    }

    return ctx;
}

extern "C" void amd_lt_destroy(void* opaque) {
    destroy_lt(static_cast<LtContext*>(opaque));
}

extern "C" int amd_lt_tune(void* opaque,
                           const float* A,
                           const float* x,
                           float* y,
                           hipStream_t stream,
                           int requested_algorithms,
                           int warmup,
                           int samples,
                           int iterations_per_sample,
                           int* returned_algorithms,
                           int* valid_algorithms,
                           int* selected_solution_index,
                           std::uint64_t* selected_workspace_bytes,
                           double* selected_median_us) {
    auto* ctx = static_cast<LtContext*>(opaque);
    if (!ctx || !A || !x || !y || requested_algorithms <= 0 ||
        warmup < 0 || samples <= 0 || iterations_per_sample <= 0) {
        return static_cast<int>(HIPBLAS_STATUS_INVALID_VALUE);
    }

    std::vector<hipblasLtMatmulHeuristicResult_t> h(
        static_cast<std::size_t>(requested_algorithms));
    int returned = 0;

    const hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(
        ctx->handle, ctx->op, ctx->a, ctx->b, ctx->c, ctx->d,
        ctx->pref, requested_algorithms, h.data(), &returned);

    if (returned_algorithms) *returned_algorithms = returned;
    if (hs != HIPBLAS_STATUS_SUCCESS || returned <= 0)
        return static_cast<int>(hs);

    std::size_t max_candidate_workspace = 0;
    for (int i = 0; i < returned; ++i) {
        if (h[i].state == HIPBLAS_STATUS_SUCCESS &&
            h[i].workspaceSize <= ctx->max_workspace_bytes) {
            max_candidate_workspace =
                std::max(max_candidate_workspace, h[i].workspaceSize);
        }
    }

    if (max_candidate_workspace > 0) {
        if (hipMalloc(&ctx->workspace, max_candidate_workspace) != hipSuccess)
            return static_cast<int>(HIPBLAS_STATUS_ALLOC_FAILED);
    }

    hipEvent_t start = nullptr, stop = nullptr;
    if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) {
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
        return static_cast<int>(HIPBLAS_STATUS_INTERNAL_ERROR);
    }

    double best = std::numeric_limits<double>::infinity();
    int best_index = -1;
    int valid = 0;

    for (int candidate = 0; candidate < returned; ++candidate) {
        if (h[candidate].state != HIPBLAS_STATUS_SUCCESS ||
            h[candidate].workspaceSize > ctx->max_workspace_bytes)
            continue;

        bool ok = true;

        for (int i = 0; i < warmup; ++i) {
            if (lt_launch_algo(ctx, &h[candidate].algo, h[candidate].workspaceSize,
                               A, x, y, stream) != HIPBLAS_STATUS_SUCCESS) {
                ok = false; break;
            }
        }
        if (ok && hipStreamSynchronize(stream) != hipSuccess) ok = false;
        if (!ok) continue;

        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(samples));

        for (int s = 0; s < samples && ok; ++s) {
            if (hipEventRecord(start, stream) != hipSuccess) { ok = false; break; }

            for (int it = 0; it < iterations_per_sample; ++it) {
                if (lt_launch_algo(ctx, &h[candidate].algo, h[candidate].workspaceSize,
                                   A, x, y, stream) != HIPBLAS_STATUS_SUCCESS) {
                    ok = false; break;
                }
            }

            if (!ok) break;

            if (hipEventRecord(stop, stream) != hipSuccess ||
                hipEventSynchronize(stop) != hipSuccess) {
                ok = false; break;
            }

            float ms = 0.0f;
            if (hipEventElapsedTime(&ms, start, stop) != hipSuccess) {
                ok = false; break;
            }

            values.push_back(
                static_cast<double>(ms) * 1000.0 /
                static_cast<double>(iterations_per_sample));
        }

        if (!ok || values.size() != static_cast<std::size_t>(samples))
            continue;

        ++valid;
        const double med = median(values);

        if (med < best) {
            best = med;
            best_index = candidate;
        }
    }

    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);

    if (valid_algorithms) *valid_algorithms = valid;
    if (best_index < 0)
        return static_cast<int>(HIPBLAS_STATUS_NOT_SUPPORTED);

    ctx->selected_algo = h[best_index].algo;
    ctx->selected_solution_index =
        hipblaslt_ext::getIndexFromAlgo(ctx->selected_algo);
    if (ctx->selected_solution_index < 0)
        return static_cast<int>(HIPBLAS_STATUS_INTERNAL_ERROR);

    ctx->selected_workspace_bytes = h[best_index].workspaceSize;
    ctx->has_selected_algo = true;

    if (selected_solution_index)
        *selected_solution_index = ctx->selected_solution_index;
    if (selected_workspace_bytes)
        *selected_workspace_bytes =
            static_cast<std::uint64_t>(ctx->selected_workspace_bytes);
    if (selected_median_us) *selected_median_us = best;

    return static_cast<int>(HIPBLAS_STATUS_SUCCESS);
}

extern "C" int amd_lt_launch(void* opaque,
                             const float* A,
                             const float* x,
                             float* y,
                             hipStream_t stream) {
    auto* ctx = static_cast<LtContext*>(opaque);
    if (!ctx || !ctx->has_selected_algo)
        return static_cast<int>(HIPBLAS_STATUS_NOT_SUPPORTED);

    return static_cast<int>(
        lt_launch_algo(ctx, &ctx->selected_algo, ctx->selected_workspace_bytes,
                       A, x, y, stream));
}

extern "C" int amd_lt_success_code() {
    return static_cast<int>(HIPBLAS_STATUS_SUCCESS);
}

extern "C" int amd_lt_load_solution_index(void* opaque, int solution_index) {
    auto* ctx = static_cast<LtContext*>(opaque);
    if (!ctx || solution_index < 0) return 0;

    std::vector<int> indices{solution_index};
    std::vector<hipblasLtMatmulHeuristicResult_t> results;

    if (hipblaslt_ext::getAlgosFromIndex(ctx->handle, indices, results)
        != HIPBLAS_STATUS_SUCCESS || results.empty()) {
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (auto& result : results) {
        if (hipblaslt_ext::getIndexFromAlgo(result.algo) != solution_index)
            continue;

        std::size_t workspace_bytes = 0;
        if (hipblaslt_ext::matmulIsAlgoSupported(
                ctx->handle,
                ctx->op,
                &alpha,
                ctx->a,
                ctx->b,
                &beta,
                ctx->c,
                ctx->d,
                result.algo,
                workspace_bytes) != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }

        if (workspace_bytes > ctx->max_workspace_bytes)
            return 0;

        if (ctx->workspace) {
            (void)hipFree(ctx->workspace);
            ctx->workspace = nullptr;
        }

        if (workspace_bytes > 0) {
            if (hipMalloc(&ctx->workspace, workspace_bytes) != hipSuccess)
                return 0;
        }

        ctx->selected_algo = result.algo;
        ctx->selected_solution_index = solution_index;
        ctx->selected_workspace_bytes = workspace_bytes;
        ctx->has_selected_algo = true;
        return 1;
    }

    return 0;
}
