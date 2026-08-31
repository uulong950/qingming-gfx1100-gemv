#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

extern "C" hipError_t gemv(
    const float* A,
    const float* x,
    float* y,
    std::int64_t m,
    std::int64_t n,
    std::int64_t lda,
    void* workspace,
    hipStream_t stream);

extern "C" std::uint64_t gemv_workspace_bytes(
    std::int64_t m,
    std::int64_t n);

extern "C" const char* gemv_path_name(
    std::int64_t m,
    std::int64_t n);
