#include "gemv.h"

#include <cstdint>
#include <cmath>
#include <limits>

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

namespace {

constexpr int kWaveSize = 32;
constexpr int kWavesPerBlock = 8;
constexpr int kBlockSize = kWaveSize * kWavesPerBlock;
constexpr int kTileK = 256;

constexpr std::uint64_t kPrefetchMinN = 2048;
constexpr std::uint64_t kPrefetchMaxMatrixBytes = 64ull * 1024ull * 1024ull;

constexpr std::uint64_t kSplitMaxM = 256;
constexpr std::uint64_t kSplitMinN = 2048;
constexpr std::uint64_t kSplitRotation3MinN = 32768;

constexpr std::uint64_t kGridSplitMaxM = 64;
constexpr std::uint64_t kGridSplitMinN = 16384;

constexpr std::uint64_t kStreamingMatrixBytes = 128ull * 1024ull * 1024ull;
constexpr std::uint64_t kStableAccumMinM = 1024;
constexpr std::uint64_t kStableAccumMinN = 65536;

static_assert(kBlockSize == 256);
static_assert(kTileK == 256);

template<int Width>
__device__ __forceinline__ float reduce_subgroup(float value)
{
    static_assert(Width == 1 || Width == 2 || Width == 4 ||
                  Width == 8 || Width == 16 || Width == 32);
#pragma unroll
    for(int offset = Width / 2; offset > 0; offset >>= 1)
        value += __shfl_down(value, offset, Width);
    return value;
}


__device__ __forceinline__ void compensated_add(
    float value,
    float& sum,
    float& correction)
{
    const float next = sum + value;
    if(fabsf(sum) >= fabsf(value))
        correction += (sum - next) + value;
    else
        correction += (value - next) + sum;
    sum = next;
}

__device__ __forceinline__ float compensated_sum8(
    const float* values)
{
    float sum = 0.0f;
    float correction = 0.0f;
#pragma unroll
    for(int i = 0; i < 8; ++i)
        compensated_add(values[i], sum, correction);
    return sum + correction;
}

__device__ __forceinline__ float compensated_sum_dynamic(
    const float* values,
    int count)
{
    float sum = 0.0f;
    float correction = 0.0f;
    for(int i = 0; i < count; ++i)
        compensated_add(values[i], sum, correction);
    return sum + correction;
}

template<int Width>
__global__ __launch_bounds__(256, 2)
void small_n_kernel(const float* __restrict__ A,
                    const float* __restrict__ x,
                    float* __restrict__ y,
                    std::uint64_t m,
                    std::uint64_t n,
                    std::uint64_t lda)
{
    constexpr int outputs_per_wave = kWaveSize / Width;

    const int thread = static_cast<int>(threadIdx.x);
    const int wave = thread >> 5;
    const int lane = thread & 31;

    const std::uint64_t global_wave =
        static_cast<std::uint64_t>(blockIdx.x) * kWavesPerBlock +
        static_cast<std::uint64_t>(wave);
    const std::uint64_t first_row =
        global_wave * static_cast<std::uint64_t>(outputs_per_wave);

    if(first_row >= m)
        return;

    const int output_slot = lane / Width;
    const int reduction_lane = lane & (Width - 1);
    const std::uint64_t row =
        first_row + static_cast<std::uint64_t>(output_slot);

    float x_source = 0.0f;
    if(lane < Width && static_cast<std::uint64_t>(lane) < n)
        x_source = x[lane];

    const float xv = __shfl(x_source, reduction_lane, kWaveSize);

    float partial = 0.0f;
    if(row < m && static_cast<std::uint64_t>(reduction_lane) < n)
    {
        partial = fmaf(
            A[row * lda + static_cast<std::uint64_t>(reduction_lane)],
            xv,
            partial);
    }

    const float sum = reduce_subgroup<Width>(partial);
    if(reduction_lane == 0 && row < m)
        y[row] = sum;
}

struct TileRegisters {
    float r0, r1, r2, r3, r4, r5, r6, r7;
};

template<bool NT>
__device__ __forceinline__ float load_matrix(const float* p)
{
#if __has_builtin(__builtin_nontemporal_load)
    if constexpr(NT)
        return __builtin_nontemporal_load(p);
#endif
    return *p;
}

template<int Width, int Rows>
__global__ __launch_bounds__(256, 2)
void pack_kernel(const float* __restrict__ A,
                 const float* __restrict__ x,
                 float* __restrict__ y,
                 std::uint64_t m,
                 std::uint64_t n,
                 std::uint64_t lda)
{
    constexpr int groups = kWaveSize / Width;
    constexpr int rows_per_wave = groups * Rows;

    const int t = static_cast<int>(threadIdx.x);
    const int wave = t >> 5;
    const int lane = t & 31;
    const int group = lane / Width;
    const int slane = lane & (Width - 1);

    const std::uint64_t gw =
        static_cast<std::uint64_t>(blockIdx.x) * kWavesPerBlock +
        static_cast<std::uint64_t>(wave);
    const std::uint64_t row0 =
        gw * static_cast<std::uint64_t>(rows_per_wave) +
        static_cast<std::uint64_t>(group * Rows);

    float acc[Rows];
#pragma unroll
    for(int r = 0; r < Rows; ++r)
        acc[r] = 0.0f;

    for(std::uint64_t col = static_cast<std::uint64_t>(slane);
        col < n;
        col += Width)
    {
        const float xv = x[col];
#pragma unroll
        for(int r = 0; r < Rows; ++r)
        {
            const std::uint64_t row = row0 + static_cast<std::uint64_t>(r);
            if(row < m)
                acc[r] = fmaf(A[row * lda + col], xv, acc[r]);
        }
    }

#pragma unroll
    for(int r = 0; r < Rows; ++r)
    {
        const float v = reduce_subgroup<Width>(acc[r]);
        const std::uint64_t row = row0 + static_cast<std::uint64_t>(r);
        if(slane == 0 && row < m)
            y[row] = v;
    }
}

template<int Width>
__global__ __launch_bounds__(256, 2)
void pack_acc2_kernel(const float* __restrict__ A,
                      const float* __restrict__ x,
                      float* __restrict__ y,
                      std::uint64_t m,
                      std::uint64_t n,
                      std::uint64_t lda)
{
    constexpr int groups = kWaveSize / Width;

    const int t = static_cast<int>(threadIdx.x);
    const int wave = t >> 5;
    const int lane = t & 31;
    const int group = lane / Width;
    const int slane = lane & (Width - 1);

    const std::uint64_t gw =
        static_cast<std::uint64_t>(blockIdx.x) * kWavesPerBlock +
        static_cast<std::uint64_t>(wave);
    const std::uint64_t row =
        gw * static_cast<std::uint64_t>(groups) +
        static_cast<std::uint64_t>(group);

    if(row >= m)
        return;

    float a0 = 0.0f;
    float a1 = 0.0f;
    int strip = 0;

    for(std::uint64_t col = static_cast<std::uint64_t>(slane);
        col < n;
        col += Width, ++strip)
    {
        const float av = A[row * lda + col];
        const float xv = x[col];

        if(strip & 1)
            a1 = fmaf(av, xv, a1);
        else
            a0 = fmaf(av, xv, a0);
    }

    float v = a0 + a1;
    v = reduce_subgroup<Width>(v);

    if(slane == 0)
        y[row] = v;
}

template<int Rows, bool NT>
__global__ __launch_bounds__(256, 2)
void rows_kernel(const float* __restrict__ A,
                 const float* __restrict__ x,
                 float* __restrict__ y,
                 std::uint64_t m,
                 std::uint64_t n,
                 std::uint64_t lda)
{
    const int t = static_cast<int>(threadIdx.x);
    const int wave = t >> 5;
    const int lane = t & 31;

    const std::uint64_t gw =
        static_cast<std::uint64_t>(blockIdx.x) * kWavesPerBlock +
        static_cast<std::uint64_t>(wave);
    const std::uint64_t row0 =
        gw * static_cast<std::uint64_t>(Rows);

    float acc[Rows];
#pragma unroll
    for(int r = 0; r < Rows; ++r)
        acc[r] = 0.0f;

    for(std::uint64_t col = static_cast<std::uint64_t>(lane);
        col < n;
        col += kWaveSize)
    {
        const float xv = x[col];
#pragma unroll
        for(int r = 0; r < Rows; ++r)
        {
            const std::uint64_t row = row0 + static_cast<std::uint64_t>(r);
            if(row < m)
                acc[r] = fmaf(
                    load_matrix<NT>(A + row * lda + col),
                    xv,
                    acc[r]);
        }
    }

#pragma unroll
    for(int r = 0; r < Rows; ++r)
    {
        const float v = reduce_subgroup<32>(acc[r]);
        const std::uint64_t row = row0 + static_cast<std::uint64_t>(r);
        if(lane == 0 && row < m)
            y[row] = v;
    }
}

template<bool NT>
__device__ __forceinline__ TileRegisters load_tile(
    const float* __restrict__ A,
    std::uint64_t row_base,
    std::uint64_t base_col,
    int physical_wave,
    int lane)
{
    const int phase = physical_wave & 7;
    const std::uint64_t l = static_cast<std::uint64_t>(lane);

    TileRegisters s;
    s.r0 = load_matrix<NT>(A + row_base + base_col + static_cast<std::uint64_t>((phase + 0) & 7) * 32u + l);
    s.r1 = load_matrix<NT>(A + row_base + base_col + static_cast<std::uint64_t>((phase + 1) & 7) * 32u + l);
    s.r2 = load_matrix<NT>(A + row_base + base_col + static_cast<std::uint64_t>((phase + 2) & 7) * 32u + l);
    s.r3 = load_matrix<NT>(A + row_base + base_col + static_cast<std::uint64_t>((phase + 3) & 7) * 32u + l);
    s.r4 = load_matrix<NT>(A + row_base + base_col + static_cast<std::uint64_t>((phase + 4) & 7) * 32u + l);
    s.r5 = load_matrix<NT>(A + row_base + base_col + static_cast<std::uint64_t>((phase + 5) & 7) * 32u + l);
    s.r6 = load_matrix<NT>(A + row_base + base_col + static_cast<std::uint64_t>((phase + 6) & 7) * 32u + l);
    s.r7 = load_matrix<NT>(A + row_base + base_col + static_cast<std::uint64_t>((phase + 7) & 7) * 32u + l);
    return s;
}

__device__ __forceinline__ void accumulate_tile(
    const TileRegisters& s,
    const float* __restrict__ x,
    std::uint64_t base_col,
    int physical_wave,
    int lane,
    float& partial)
{
    const int phase = physical_wave & 7;
    const std::uint64_t l = static_cast<std::uint64_t>(lane);

    const float x0 = x[base_col + 0u * 32u + l];
    const float x1 = x[base_col + 1u * 32u + l];
    const float x2 = x[base_col + 2u * 32u + l];
    const float x3 = x[base_col + 3u * 32u + l];
    const float x4 = x[base_col + 4u * 32u + l];
    const float x5 = x[base_col + 5u * 32u + l];
    const float x6 = x[base_col + 6u * 32u + l];
    const float x7 = x[base_col + 7u * 32u + l];

#define APPLY_FMA8(a0,a1,a2,a3,a4,a5,a6,a7) \
    do { \
        partial = fmaf((a0), x0, partial); \
        partial = fmaf((a1), x1, partial); \
        partial = fmaf((a2), x2, partial); \
        partial = fmaf((a3), x3, partial); \
        partial = fmaf((a4), x4, partial); \
        partial = fmaf((a5), x5, partial); \
        partial = fmaf((a6), x6, partial); \
        partial = fmaf((a7), x7, partial); \
    } while(0)

    switch(phase)
    {
        case 0: APPLY_FMA8(s.r0,s.r1,s.r2,s.r3,s.r4,s.r5,s.r6,s.r7); break;
        case 1: APPLY_FMA8(s.r7,s.r0,s.r1,s.r2,s.r3,s.r4,s.r5,s.r6); break;
        case 2: APPLY_FMA8(s.r6,s.r7,s.r0,s.r1,s.r2,s.r3,s.r4,s.r5); break;
        case 3: APPLY_FMA8(s.r5,s.r6,s.r7,s.r0,s.r1,s.r2,s.r3,s.r4); break;
        case 4: APPLY_FMA8(s.r4,s.r5,s.r6,s.r7,s.r0,s.r1,s.r2,s.r3); break;
        case 5: APPLY_FMA8(s.r3,s.r4,s.r5,s.r6,s.r7,s.r0,s.r1,s.r2); break;
        case 6: APPLY_FMA8(s.r2,s.r3,s.r4,s.r5,s.r6,s.r7,s.r0,s.r1); break;
        default:APPLY_FMA8(s.r1,s.r2,s.r3,s.r4,s.r5,s.r6,s.r7,s.r0); break;
    }
#undef APPLY_FMA8
}

__device__ __forceinline__ void accumulate_tile_four_way(
    const TileRegisters& s,
    const float* __restrict__ x,
    std::uint64_t base_col,
    int physical_wave,
    int lane,
    float& p0,
    float& p1,
    float& p2,
    float& p3)
{
    const int phase = physical_wave & 7;
    const std::uint64_t l = static_cast<std::uint64_t>(lane);

    const float x0 = x[base_col + 0u * 32u + l];
    const float x1 = x[base_col + 1u * 32u + l];
    const float x2 = x[base_col + 2u * 32u + l];
    const float x3 = x[base_col + 3u * 32u + l];
    const float x4 = x[base_col + 4u * 32u + l];
    const float x5 = x[base_col + 5u * 32u + l];
    const float x6 = x[base_col + 6u * 32u + l];
    const float x7 = x[base_col + 7u * 32u + l];

#define APPLY_FMA8_4(a0,a1,a2,a3,a4,a5,a6,a7) \
    do { \
        p0 = fmaf((a0), x0, p0); \
        p1 = fmaf((a1), x1, p1); \
        p2 = fmaf((a2), x2, p2); \
        p3 = fmaf((a3), x3, p3); \
        p0 = fmaf((a4), x4, p0); \
        p1 = fmaf((a5), x5, p1); \
        p2 = fmaf((a6), x6, p2); \
        p3 = fmaf((a7), x7, p3); \
    } while(0)

    switch(phase)
    {
        case 0: APPLY_FMA8_4(s.r0,s.r1,s.r2,s.r3,s.r4,s.r5,s.r6,s.r7); break;
        case 1: APPLY_FMA8_4(s.r7,s.r0,s.r1,s.r2,s.r3,s.r4,s.r5,s.r6); break;
        case 2: APPLY_FMA8_4(s.r6,s.r7,s.r0,s.r1,s.r2,s.r3,s.r4,s.r5); break;
        case 3: APPLY_FMA8_4(s.r5,s.r6,s.r7,s.r0,s.r1,s.r2,s.r3,s.r4); break;
        case 4: APPLY_FMA8_4(s.r4,s.r5,s.r6,s.r7,s.r0,s.r1,s.r2,s.r3); break;
        case 5: APPLY_FMA8_4(s.r3,s.r4,s.r5,s.r6,s.r7,s.r0,s.r1,s.r2); break;
        case 6: APPLY_FMA8_4(s.r2,s.r3,s.r4,s.r5,s.r6,s.r7,s.r0,s.r1); break;
        default:APPLY_FMA8_4(s.r1,s.r2,s.r3,s.r4,s.r5,s.r6,s.r7,s.r0); break;
    }
#undef APPLY_FMA8_4
}

template<bool NT>
__device__ __forceinline__ void accumulate_tail(
    const float* __restrict__ A,
    const float* __restrict__ x,
    std::uint64_t row_base,
    std::uint64_t tail_base,
    std::uint64_t n,
    int lane,
    float& partial)
{
    if constexpr(NT)
    {
#pragma unroll
        for(int strip = 0; strip < 8; ++strip)
        {
            const std::uint64_t col =
                tail_base + static_cast<std::uint64_t>(strip * kWaveSize + lane);
            if(col < n)
                partial = fmaf(load_matrix<NT>(A + row_base + col), x[col], partial);
        }
    }
    else
    {
        for(std::uint64_t col = tail_base + static_cast<std::uint64_t>(lane);
            col < n;
            col += kWaveSize)
        {
            partial = fmaf(A[row_base + col], x[col], partial);
        }
    }
}

template<bool NT, bool Acc4, bool Prefetch>
__global__ __launch_bounds__(256, 2)
void row_kernel(const float* __restrict__ A,
                const float* __restrict__ x,
                float* __restrict__ y,
                std::uint64_t m,
                std::uint64_t n,
                std::uint64_t lda)
{
    static_assert(!(Prefetch && Acc4));

    const int thread = static_cast<int>(threadIdx.x);
    const int physical_wave = thread >> 5;
    const int lane = thread & 31;
    const std::uint64_t row =
        static_cast<std::uint64_t>(blockIdx.x) * kWavesPerBlock +
        static_cast<std::uint64_t>(physical_wave);

    if(row >= m)
        return;

    const std::uint64_t row_base = row * lda;
    const std::uint64_t full_tiles = n / kTileK;

    if constexpr(Acc4)
    {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;

        for(std::uint64_t t = 0; t < full_tiles; ++t)
        {
            const std::uint64_t base_col = t * kTileK;
            const TileRegisters tile =
                load_tile<NT>(A, row_base, base_col, physical_wave, lane);
            accumulate_tile_four_way(
                tile, x, base_col, physical_wave, lane, p0, p1, p2, p3);
        }

        float partial = (p0 + p1) + (p2 + p3);
        accumulate_tail<NT>(
            A, x, row_base, full_tiles * kTileK, n, lane, partial);

        const float sum = reduce_subgroup<32>(partial);
        if(lane == 0)
            y[row] = sum;
    }
    else
    {
        float partial = 0.0f;

        if constexpr(Prefetch)
        {
            if(full_tiles > 0)
            {
                std::uint64_t current_base = 0;
                TileRegisters current =
                    load_tile<NT>(A, row_base, current_base, physical_wave, lane);

                for(std::uint64_t next_t = 1; next_t < full_tiles; ++next_t)
                {
                    const std::uint64_t next_base = next_t * kTileK;
                    const TileRegisters next =
                        load_tile<NT>(A, row_base, next_base, physical_wave, lane);

                    accumulate_tile(
                        current, x, current_base, physical_wave, lane, partial);
                    current = next;
                    current_base = next_base;
                }

                accumulate_tile(
                    current, x, current_base, physical_wave, lane, partial);
            }
        }
        else
        {
            for(std::uint64_t t = 0; t < full_tiles; ++t)
            {
                const std::uint64_t base_col = t * kTileK;
                const TileRegisters tile =
                    load_tile<NT>(A, row_base, base_col, physical_wave, lane);
                accumulate_tile(
                    tile, x, base_col, physical_wave, lane, partial);
            }
        }

        accumulate_tail<NT>(
            A, x, row_base, full_tiles * kTileK, n, lane, partial);

        const float sum = reduce_subgroup<32>(partial);
        if(lane == 0)
            y[row] = sum;
    }
}

template<int Wave>
__device__ __forceinline__ void accumulate_row4096_static(
    const float* __restrict__ A,
    const float* __restrict__ x,
    std::uint64_t row_base,
    int lane,
    float& partial)
{
    static_assert(Wave >= 0 && Wave < kWavesPerBlock);

#define STATIC_4096_TILE(BASE) \
    do { \
        constexpr std::uint64_t base_col = \
            static_cast<std::uint64_t>(BASE); \
        const TileRegisters tile = \
            load_tile<false>(A, row_base, base_col, Wave, lane); \
        accumulate_tile( \
            tile, x, base_col, Wave, lane, partial); \
    } while(0)

    STATIC_4096_TILE(0);
    STATIC_4096_TILE(256);
    STATIC_4096_TILE(512);
    STATIC_4096_TILE(768);
    STATIC_4096_TILE(1024);
    STATIC_4096_TILE(1280);
    STATIC_4096_TILE(1536);
    STATIC_4096_TILE(1792);
    STATIC_4096_TILE(2048);
    STATIC_4096_TILE(2304);
    STATIC_4096_TILE(2560);
    STATIC_4096_TILE(2816);
    STATIC_4096_TILE(3072);
    STATIC_4096_TILE(3328);
    STATIC_4096_TILE(3584);
    STATIC_4096_TILE(3840);

#undef STATIC_4096_TILE
}

__global__ __launch_bounds__(256, 2)
void row4096_static_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    std::uint64_t m,
    std::uint64_t lda)
{
    const int thread = static_cast<int>(threadIdx.x);
    const int wave = thread >> 5;
    const int lane = thread & 31;

    const std::uint64_t row =
        static_cast<std::uint64_t>(blockIdx.x) * kWavesPerBlock +
        static_cast<std::uint64_t>(wave);

    if(row >= m)
        return;

    const std::uint64_t row_base = row * lda;
    float partial = 0.0f;

    switch(wave)
    {
        case 0:
            accumulate_row4096_static<0>(
                A, x, row_base, lane, partial);
            break;
        case 1:
            accumulate_row4096_static<1>(
                A, x, row_base, lane, partial);
            break;
        case 2:
            accumulate_row4096_static<2>(
                A, x, row_base, lane, partial);
            break;
        case 3:
            accumulate_row4096_static<3>(
                A, x, row_base, lane, partial);
            break;
        case 4:
            accumulate_row4096_static<4>(
                A, x, row_base, lane, partial);
            break;
        case 5:
            accumulate_row4096_static<5>(
                A, x, row_base, lane, partial);
            break;
        case 6:
            accumulate_row4096_static<6>(
                A, x, row_base, lane, partial);
            break;
        default:
            accumulate_row4096_static<7>(
                A, x, row_base, lane, partial);
            break;
    }

    const float sum = reduce_subgroup<32>(partial);

    if(lane == 0)
        y[row] = sum;
}

constexpr unsigned int kRow4095PersistentBlocks = 384u;
constexpr unsigned int kRow4097PersistentBlocks = 384u;

template<int Strip>
__device__ __forceinline__ float load_row4095_strip(
    const float* __restrict__ A,
    std::uint64_t row_base,
    int lane)
{
    static_assert(Strip >= 0 && Strip < 8);

    if constexpr(Strip < 7)
    {
        return load_matrix<true>(
            A + row_base + 3840u +
            static_cast<std::uint64_t>(Strip * kWaveSize + lane));
    }
    else
    {

        const int safe_lane = lane < 31 ? lane : 30;
        return load_matrix<true>(
            A + row_base + 3840u + 7u * kWaveSize +
            static_cast<std::uint64_t>(safe_lane));
    }
}

template<int Wave>
__device__ __forceinline__ TileRegisters load_row4095_tile15(
    const float* __restrict__ A,
    std::uint64_t row_base,
    int lane)
{
    static_assert(Wave >= 0 && Wave < kWavesPerBlock);
    constexpr int phase = Wave & 7;

    TileRegisters r;
    r.r0 = load_row4095_strip<(phase + 0) & 7>(A, row_base, lane);
    r.r1 = load_row4095_strip<(phase + 1) & 7>(A, row_base, lane);
    r.r2 = load_row4095_strip<(phase + 2) & 7>(A, row_base, lane);
    r.r3 = load_row4095_strip<(phase + 3) & 7>(A, row_base, lane);
    r.r4 = load_row4095_strip<(phase + 4) & 7>(A, row_base, lane);
    r.r5 = load_row4095_strip<(phase + 5) & 7>(A, row_base, lane);
    r.r6 = load_row4095_strip<(phase + 6) & 7>(A, row_base, lane);
    r.r7 = load_row4095_strip<(phase + 7) & 7>(A, row_base, lane);
    return r;
}

template<int Wave>
__device__ __forceinline__ void accumulate_row4095_a2(
    const TileRegisters& r,
    const float* __restrict__ q,
    float& p0,
    float& p1)
{
    static_assert(Wave >= 0 && Wave < kWavesPerBlock);

    const float x0 = q[0];
    const float x1 = q[1];
    const float x2 = q[2];
    const float x3 = q[3];
    const float x4 = q[4];
    const float x5 = q[5];
    const float x6 = q[6];
    const float x7 = q[7];

#define APPLY_PERSIST_A2(a0,a1,a2,a3,a4,a5,a6,a7) \
    do { \
        p0 = fmaf((a0), x0, p0); \
        p1 = fmaf((a1), x1, p1); \
        p0 = fmaf((a2), x2, p0); \
        p1 = fmaf((a3), x3, p1); \
        p0 = fmaf((a4), x4, p0); \
        p1 = fmaf((a5), x5, p1); \
        p0 = fmaf((a6), x6, p0); \
        p1 = fmaf((a7), x7, p1); \
    } while(0)

    constexpr int phase = Wave & 7;
    if constexpr(phase == 0)
        APPLY_PERSIST_A2(r.r0,r.r1,r.r2,r.r3,r.r4,r.r5,r.r6,r.r7);
    else if constexpr(phase == 1)
        APPLY_PERSIST_A2(r.r7,r.r0,r.r1,r.r2,r.r3,r.r4,r.r5,r.r6);
    else if constexpr(phase == 2)
        APPLY_PERSIST_A2(r.r6,r.r7,r.r0,r.r1,r.r2,r.r3,r.r4,r.r5);
    else if constexpr(phase == 3)
        APPLY_PERSIST_A2(r.r5,r.r6,r.r7,r.r0,r.r1,r.r2,r.r3,r.r4);
    else if constexpr(phase == 4)
        APPLY_PERSIST_A2(r.r4,r.r5,r.r6,r.r7,r.r0,r.r1,r.r2,r.r3);
    else if constexpr(phase == 5)
        APPLY_PERSIST_A2(r.r3,r.r4,r.r5,r.r6,r.r7,r.r0,r.r1,r.r2);
    else if constexpr(phase == 6)
        APPLY_PERSIST_A2(r.r2,r.r3,r.r4,r.r5,r.r6,r.r7,r.r0,r.r1);
    else
        APPLY_PERSIST_A2(r.r1,r.r2,r.r3,r.r4,r.r5,r.r6,r.r7,r.r0);

#undef APPLY_PERSIST_A2
}

template<int Tile>
__device__ __forceinline__ void load_row4095_x_tile(
    const float* __restrict__ x,
    int lane,
    float* __restrict__ q)
{
    static_assert(Tile >= 0 && Tile <= 15);

#pragma unroll
    for(int strip = 0; strip < 8; ++strip)
    {
        const int col =
            Tile * kTileK + strip * kWaveSize + lane;

        if constexpr(Tile < 15)
            q[strip] = x[col];
        else
            q[strip] = col < 4095 ? x[col] : 0.0f;
    }
}

template<int Wave>
__device__ __forceinline__ void load_row4095_x_partition(
    const float* __restrict__ x,
    int lane,
    float* __restrict__ q)
{
    static_assert(Wave >= 0 && Wave < kWavesPerBlock);
    constexpr int part = Wave & 3;

    load_row4095_x_tile<part + 0 * 4>(x, lane, q + 0);
    load_row4095_x_tile<part + 1 * 4>(x, lane, q + 8);
    load_row4095_x_tile<part + 2 * 4>(x, lane, q + 16);
    load_row4095_x_tile<part + 3 * 4>(x, lane, q + 24);
}

template<int Wave>
__device__ __forceinline__ void accumulate_row4095_partition(
    const float* __restrict__ A,
    const float* __restrict__ q,
    std::uint64_t row_base,
    int lane,
    float& p0,
    float& p1)
{
    static_assert(Wave >= 0 && Wave < kWavesPerBlock);
    constexpr int part = Wave & 3;

    constexpr std::uint64_t b0 =
        static_cast<std::uint64_t>(part + 0 * 4) * kTileK;
    constexpr std::uint64_t b1 =
        static_cast<std::uint64_t>(part + 1 * 4) * kTileK;
    constexpr std::uint64_t b2 =
        static_cast<std::uint64_t>(part + 2 * 4) * kTileK;
    constexpr std::uint64_t b3 =
        static_cast<std::uint64_t>(part + 3 * 4) * kTileK;

    const TileRegisters r0 =
        load_tile<true>(A, row_base, b0, Wave, lane);
    accumulate_row4095_a2<Wave>(r0, q + 0, p0, p1);

    const TileRegisters r1 =
        load_tile<true>(A, row_base, b1, Wave, lane);
    accumulate_row4095_a2<Wave>(r1, q + 8, p0, p1);

    const TileRegisters r2 =
        load_tile<true>(A, row_base, b2, Wave, lane);
    accumulate_row4095_a2<Wave>(r2, q + 16, p0, p1);

    if constexpr(part != 3)
    {
        const TileRegisters r3 =
            load_tile<true>(A, row_base, b3, Wave, lane);
        accumulate_row4095_a2<Wave>(r3, q + 24, p0, p1);
    }
    else
    {
        const TileRegisters r3 =
            load_row4095_tile15<Wave>(A, row_base, lane);
        accumulate_row4095_a2<Wave>(r3, q + 24, p0, p1);
    }
}

__global__ __launch_bounds__(256, 2)
void row4095_persistent_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda)
{
    __shared__ float sums[2][kWavesPerBlock];

    const int t = static_cast<int>(threadIdx.x);
    const int wave = t >> 5;
    const int lane = t & 31;
    const int row_in_pair = wave >> 2;
    const int part = wave & 3;

    float q[32];

    switch(wave)
    {
        case 0: load_row4095_x_partition<0>(x, lane, q); break;
        case 1: load_row4095_x_partition<1>(x, lane, q); break;
        case 2: load_row4095_x_partition<2>(x, lane, q); break;
        case 3: load_row4095_x_partition<3>(x, lane, q); break;
        case 4: load_row4095_x_partition<4>(x, lane, q); break;
        case 5: load_row4095_x_partition<5>(x, lane, q); break;
        case 6: load_row4095_x_partition<6>(x, lane, q); break;
        default:load_row4095_x_partition<7>(x, lane, q); break;
    }

    const std::uint64_t total_pairs = (m + 1u) / 2u;
    const std::uint64_t pairs_per_block =
        (total_pairs +
         static_cast<std::uint64_t>(kRow4095PersistentBlocks) - 1u) /
        static_cast<std::uint64_t>(kRow4095PersistentBlocks);

    const std::uint64_t physical =
        static_cast<std::uint64_t>(blockIdx.x);

    const std::uint64_t logical =
        static_cast<std::uint64_t>(kRow4095PersistentBlocks) - 1u - physical;

    const std::uint64_t begin = logical * pairs_per_block;
    const std::uint64_t raw_end = begin + pairs_per_block;
    const std::uint64_t end =
        raw_end < total_pairs ? raw_end : total_pairs;

    for(std::uint64_t pair = begin; pair < end; ++pair)
    {
        const std::uint64_t row =
            pair * 2u + static_cast<std::uint64_t>(row_in_pair);

        float p0 = 0.0f;
        float p1 = 0.0f;

        if(row < m)
        {
            const std::uint64_t row_base = row * lda;

            switch(wave)
            {
                case 0:
                    accumulate_row4095_partition<0>(
                        A,q,row_base,lane,p0,p1); break;
                case 1:
                    accumulate_row4095_partition<1>(
                        A,q,row_base,lane,p0,p1); break;
                case 2:
                    accumulate_row4095_partition<2>(
                        A,q,row_base,lane,p0,p1); break;
                case 3:
                    accumulate_row4095_partition<3>(
                        A,q,row_base,lane,p0,p1); break;
                case 4:
                    accumulate_row4095_partition<4>(
                        A,q,row_base,lane,p0,p1); break;
                case 5:
                    accumulate_row4095_partition<5>(
                        A,q,row_base,lane,p0,p1); break;
                case 6:
                    accumulate_row4095_partition<6>(
                        A,q,row_base,lane,p0,p1); break;
                default:
                    accumulate_row4095_partition<7>(
                        A,q,row_base,lane,p0,p1); break;
            }
        }

        float v = reduce_subgroup<32>(p0 + p1);
        const int bank = static_cast<int>(pair & 1u);

        if(lane == 0)
            sums[bank][wave] = v;

        __syncthreads();

        if(row < m && part == 0 && lane == 0)
        {
            const int base = row_in_pair * 4;
            y[row] =
                ((sums[bank][base] + sums[bank][base + 1]) +
                 sums[bank][base + 2]) +
                sums[bank][base + 3];
        }
    }
}


template<int Tile>
__device__ __forceinline__ void load_row4097_x_tile(
    const float* __restrict__ x,
    int lane,
    float* __restrict__ q)
{
    static_assert(Tile >= 0 && Tile < 16);

#pragma unroll
    for(int strip = 0; strip < 8; ++strip)
    {
        const int col =
            Tile * kTileK + strip * kWaveSize + lane;
        q[strip] = x[col];
    }
}

template<int Wave>
__device__ __forceinline__ void load_row4097_x_partition(
    const float* __restrict__ x,
    int lane,
    float* __restrict__ q)
{
    static_assert(Wave >= 0 && Wave < kWavesPerBlock);
    constexpr int part = Wave & 3;

    load_row4097_x_tile<part + 0 * 4>(x, lane, q + 0);
    load_row4097_x_tile<part + 1 * 4>(x, lane, q + 8);
    load_row4097_x_tile<part + 2 * 4>(x, lane, q + 16);
    load_row4097_x_tile<part + 3 * 4>(x, lane, q + 24);
}

template<int Wave>
__device__ __forceinline__ void accumulate_row4097_partition(
    const float* __restrict__ A,
    const float* __restrict__ q,
    float x_tail,
    std::uint64_t row_base,
    int lane,
    float& p0,
    float& p1)
{
    static_assert(Wave >= 0 && Wave < kWavesPerBlock);
    constexpr int part = Wave & 3;

    constexpr std::uint64_t b0 =
        static_cast<std::uint64_t>(part + 0 * 4) * kTileK;
    constexpr std::uint64_t b1 =
        static_cast<std::uint64_t>(part + 1 * 4) * kTileK;
    constexpr std::uint64_t b2 =
        static_cast<std::uint64_t>(part + 2 * 4) * kTileK;
    constexpr std::uint64_t b3 =
        static_cast<std::uint64_t>(part + 3 * 4) * kTileK;

    const TileRegisters r0 =
        load_tile<true>(A, row_base, b0, Wave, lane);
    accumulate_row4095_a2<Wave>(r0, q + 0, p0, p1);

    const TileRegisters r1 =
        load_tile<true>(A, row_base, b1, Wave, lane);
    accumulate_row4095_a2<Wave>(r1, q + 8, p0, p1);

    const TileRegisters r2 =
        load_tile<true>(A, row_base, b2, Wave, lane);
    accumulate_row4095_a2<Wave>(r2, q + 16, p0, p1);

    const TileRegisters r3 =
        load_tile<true>(A, row_base, b3, Wave, lane);
    accumulate_row4095_a2<Wave>(r3, q + 24, p0, p1);

    if constexpr(part == 3)
    {
        if(lane == 0)
            p0 = fmaf(
                load_matrix<true>(A + row_base + 4096u),
                x_tail,
                p0);
    }
}

__global__ __launch_bounds__(256, 2)
void row4097_persistent_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda)
{
    __shared__ float sums[2][kWavesPerBlock];

    const int t = static_cast<int>(threadIdx.x);
    const int wave = t >> 5;
    const int lane = t & 31;
    const int row_in_pair = wave >> 2;
    const int part = wave & 3;

    float q[32];

    switch(wave)
    {
        case 0: load_row4097_x_partition<0>(x, lane, q); break;
        case 1: load_row4097_x_partition<1>(x, lane, q); break;
        case 2: load_row4097_x_partition<2>(x, lane, q); break;
        case 3: load_row4097_x_partition<3>(x, lane, q); break;
        case 4: load_row4097_x_partition<4>(x, lane, q); break;
        case 5: load_row4097_x_partition<5>(x, lane, q); break;
        case 6: load_row4097_x_partition<6>(x, lane, q); break;
        default:load_row4097_x_partition<7>(x, lane, q); break;
    }

    const float x_tail = x[4096];

    const std::uint64_t total_pairs = (m + 1u) / 2u;
    const std::uint64_t pairs_per_block =
        (total_pairs +
         static_cast<std::uint64_t>(kRow4097PersistentBlocks) - 1u) /
        static_cast<std::uint64_t>(kRow4097PersistentBlocks);

    const std::uint64_t physical =
        static_cast<std::uint64_t>(blockIdx.x);

    const std::uint64_t logical =
        static_cast<std::uint64_t>(kRow4097PersistentBlocks) - 1u - physical;

    const std::uint64_t begin = logical * pairs_per_block;
    const std::uint64_t raw_end = begin + pairs_per_block;
    const std::uint64_t end =
        raw_end < total_pairs ? raw_end : total_pairs;

    for(std::uint64_t pair = begin; pair < end; ++pair)
    {
        const std::uint64_t row =
            pair * 2u + static_cast<std::uint64_t>(row_in_pair);

        float p0 = 0.0f;
        float p1 = 0.0f;

        if(row < m)
        {
            const std::uint64_t row_base = row * lda;

            switch(wave)
            {
                case 0:
                    accumulate_row4097_partition<0>(
                        A,q,x_tail,row_base,lane,p0,p1); break;
                case 1:
                    accumulate_row4097_partition<1>(
                        A,q,x_tail,row_base,lane,p0,p1); break;
                case 2:
                    accumulate_row4097_partition<2>(
                        A,q,x_tail,row_base,lane,p0,p1); break;
                case 3:
                    accumulate_row4097_partition<3>(
                        A,q,x_tail,row_base,lane,p0,p1); break;
                case 4:
                    accumulate_row4097_partition<4>(
                        A,q,x_tail,row_base,lane,p0,p1); break;
                case 5:
                    accumulate_row4097_partition<5>(
                        A,q,x_tail,row_base,lane,p0,p1); break;
                case 6:
                    accumulate_row4097_partition<6>(
                        A,q,x_tail,row_base,lane,p0,p1); break;
                default:
                    accumulate_row4097_partition<7>(
                        A,q,x_tail,row_base,lane,p0,p1); break;
            }
        }

        float v = reduce_subgroup<32>(p0 + p1);
        const int bank = static_cast<int>(pair & 1u);

        if(lane == 0)
            sums[bank][wave] = v;

        __syncthreads();

        if(row < m && part == 0 && lane == 0)
        {
            const int base = row_in_pair * 4;
            y[row] =
                ((sums[bank][base] + sums[bank][base + 1]) +
                 sums[bank][base + 2]) +
                sums[bank][base + 3];
        }
    }
}

__global__ __launch_bounds__(256, 2)
void split2_kernel(const float* __restrict__ A,
                    const float* __restrict__ x,
                    float* __restrict__ y,
                    std::uint64_t m,
                    std::uint64_t n,
                    std::uint64_t lda)
{
    __shared__ float sx[4095];
    __shared__ float sums[kWavesPerBlock];

    const int t = static_cast<int>(threadIdx.x);

    for(int col = t; col < 4095; col += kBlockSize)
        sx[col] = x[col];
    __syncthreads();

    const int wave = t >> 5;
    const int lane = t & 31;
    const int pair = wave >> 1;
    const int part = wave & 1;

    const std::uint64_t row =
        static_cast<std::uint64_t>(blockIdx.x) * 4u +
        static_cast<std::uint64_t>(pair);

    float p0 = 0.0f;
    float p1 = 0.0f;
    float p2 = 0.0f;
    float p3 = 0.0f;

    if(row < m)
    {
        const std::uint64_t rb = row * lda;
        const std::uint64_t tiles = n / kTileK;

        for(std::uint64_t ti = static_cast<std::uint64_t>(part);
            ti < tiles;
            ti += 2)
        {
            const std::uint64_t b = ti * kTileK;
            const TileRegisters tile =
                load_tile<true>(A, rb, b, wave, lane);

            accumulate_tile_four_way(
                tile, sx, b, wave, lane,
                p0, p1, p2, p3);
        }

        if(part == 1)
        {
#pragma unroll
            for(int strip = 0; strip < 8; ++strip)
            {
                const std::uint64_t col =
                    tiles * kTileK +
                    static_cast<std::uint64_t>(
                        strip * kWaveSize + lane);

                if(col < n)
                {
                    const float av = A[rb + col];
                    const float xv = sx[col];

                    switch(strip & 3)
                    {
                        case 0: p0 = fmaf(av, xv, p0); break;
                        case 1: p1 = fmaf(av, xv, p1); break;
                        case 2: p2 = fmaf(av, xv, p2); break;
                        default:p3 = fmaf(av, xv, p3); break;
                    }
                }
            }
        }
    }

    float v = (p0 + p1) + (p2 + p3);
    v = reduce_subgroup<32>(v);

    if(lane == 0)
        sums[wave] = v;

    __syncthreads();

    if(row < m && part == 0 && lane == 0)
        y[row] = sums[wave] + sums[wave + 1];
}

__global__ __launch_bounds__(256, 2)
void split4_lds_kernel(const float* __restrict__ A,
                        const float* __restrict__ x,
                        float* __restrict__ y,
                        std::uint64_t m,
                        std::uint64_t n,
                        std::uint64_t lda)
{
    __shared__ float4 sx4[1025];
    __shared__ float sums[kWavesPerBlock];

    float* sx = reinterpret_cast<float*>(sx4);
    const int t = static_cast<int>(threadIdx.x);

    for(int c = t; c < 4097; c += kBlockSize)
        sx[c] = x[c];
    __syncthreads();

    const int wave = t >> 5;
    const int lane = t & 31;
    const int row_in_block = wave >> 2;
    const int part = wave & 3;
    const std::uint64_t row =
        static_cast<std::uint64_t>(blockIdx.x) * 2u +
        static_cast<std::uint64_t>(row_in_block);

    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;

    if(row < m)
    {
        const std::uint64_t rb = row * lda;
        const std::uint64_t tiles = n / kTileK;

        for(std::uint64_t ti = static_cast<std::uint64_t>(part);
            ti < tiles;
            ti += 4)
        {
            const std::uint64_t b = ti * kTileK;
            const TileRegisters tile =
                load_tile<true>(A, rb, b, wave, lane);
            accumulate_tile_four_way(
                tile, sx, b, wave, lane, p0, p1, p2, p3);
        }

        if(part == 3)
        {
#pragma unroll
            for(int strip = 0; strip < 8; ++strip)
            {
                const std::uint64_t col =
                    tiles * kTileK +
                    static_cast<std::uint64_t>(strip * kWaveSize + lane);
                if(col < n)
                {
                    const float a = A[rb + col];
                    switch(strip & 3)
                    {
                        case 0: p0 = fmaf(a, sx[col], p0); break;
                        case 1: p1 = fmaf(a, sx[col], p1); break;
                        case 2: p2 = fmaf(a, sx[col], p2); break;
                        default:p3 = fmaf(a, sx[col], p3); break;
                    }
                }
            }
        }
    }

    float v = (p0 + p1) + (p2 + p3);
    v = reduce_subgroup<32>(v);

    if(lane == 0)
        sums[wave] = v;
    __syncthreads();

    if(row < m && part == 0 && lane == 0)
    {
        const int b = row_in_block * 4;
        y[row] = ((sums[b] + sums[b + 1]) + sums[b + 2]) + sums[b + 3];
    }
}

template<int Rotation>
__global__ __launch_bounds__(256, 2)
void split4_kernel(const float* __restrict__ A,
                   const float* __restrict__ x,
                   float* __restrict__ y,
                   std::uint64_t m,
                   std::uint64_t n,
                   std::uint64_t lda)
{
    static_assert(Rotation == 1 || Rotation == 3);
    constexpr int split = 4;
    constexpr int outputs_per_block = kWavesPerBlock / split;

    __shared__ float logical_partials[kWavesPerBlock];

    const int thread = static_cast<int>(threadIdx.x);
    const int physical_wave = thread >> 5;
    const int lane = thread & 31;
    const int output_group = physical_wave / split;
    const int physical_local_wave = physical_wave & 3;
    const int logical_slot = (physical_local_wave + Rotation) & 3;

    const std::uint64_t row =
        static_cast<std::uint64_t>(blockIdx.x) * outputs_per_block +
        static_cast<std::uint64_t>(output_group);
    const bool active = row < m;
    const std::uint64_t row_base = active ? row * lda : 0;
    const std::uint64_t full_tiles = n / kTileK;

    float partial = 0.0f;

    if(active && static_cast<std::uint64_t>(logical_slot) < full_tiles)
    {
        std::uint64_t current_t = static_cast<std::uint64_t>(logical_slot);
        std::uint64_t current_base = current_t * kTileK;
        TileRegisters current =
            load_tile<false>(
                A, row_base, current_base, physical_wave, lane);

        for(std::uint64_t next_t = current_t + split;
            next_t < full_tiles;
            next_t += split)
        {
            const std::uint64_t next_base = next_t * kTileK;
            const TileRegisters next =
                load_tile<false>(
                    A, row_base, next_base, physical_wave, lane);
            accumulate_tile(
                current, x, current_base, physical_wave, lane, partial);
            current = next;
            current_base = next_base;
        }

        accumulate_tile(
            current, x, current_base, physical_wave, lane, partial);
    }

    const std::uint64_t tail_base = full_tiles * kTileK;
    const int tail_slot = static_cast<int>(full_tiles & 3u);
    if(active && logical_slot == tail_slot)
    {
        for(std::uint64_t col = tail_base + static_cast<std::uint64_t>(lane);
            col < n;
            col += kWaveSize)
        {
            partial = fmaf(A[row_base + col], x[col], partial);
        }
    }

    const float wave_sum = reduce_subgroup<32>(partial);
    if(lane == 0)
        logical_partials[output_group * split + logical_slot] =
            active ? wave_sum : 0.0f;

    __syncthreads();

    if(active && physical_local_wave == 0 && lane == 0)
    {
        const int base = output_group * split;
        float total = logical_partials[base + 0];
        total += logical_partials[base + 1];
        total += logical_partials[base + 2];
        total += logical_partials[base + 3];
        y[row] = total;
    }
}

template<int TilesPerGroup>
__global__ __launch_bounds__(256, 2)
void grid_split_kernel(const float* __restrict__ A,
                       const float* __restrict__ x,
                       float* __restrict__ y,
                       float* __restrict__ block_partials,
                       unsigned int* __restrict__ row_counters,
                       std::uint64_t m,
                       std::uint64_t n,
                       std::uint64_t lda,
                       int groups)
{
    static_assert(TilesPerGroup == 8 || TilesPerGroup == 16 ||
                  TilesPerGroup == 32 || TilesPerGroup == 64);

    __shared__ float wave_partials[kWavesPerBlock];

    const int thread = static_cast<int>(threadIdx.x);
    const int physical_wave = thread >> 5;
    const int lane = thread & 31;
    const std::uint64_t linear_group = static_cast<std::uint64_t>(blockIdx.x);
    const std::uint64_t row =
        linear_group / static_cast<std::uint64_t>(groups);
    const int logical_group = static_cast<int>(
        linear_group - row * static_cast<std::uint64_t>(groups));

    const std::uint64_t row_base = row * lda;
    const std::uint64_t full_tiles = n / kTileK;
    const std::uint64_t begin =
        static_cast<std::uint64_t>(logical_group) * TilesPerGroup;
    const std::uint64_t raw_end = begin + TilesPerGroup;
    const std::uint64_t end = raw_end < full_tiles ? raw_end : full_tiles;

    float partial = 0.0f;
    const std::uint64_t first_t =
        begin + static_cast<std::uint64_t>(physical_wave);

    if(first_t < end)
    {
        std::uint64_t current_t = first_t;
        std::uint64_t current_base = current_t * kTileK;
        TileRegisters current =
            load_tile<false>(
                A, row_base, current_base, physical_wave, lane);

        for(std::uint64_t next_t = current_t + kWavesPerBlock;
            next_t < end;
            next_t += kWavesPerBlock)
        {
            const std::uint64_t next_base = next_t * kTileK;
            const TileRegisters next =
                load_tile<false>(
                    A, row_base, next_base, physical_wave, lane);
            accumulate_tile(
                current, x, current_base, physical_wave, lane, partial);
            current = next;
            current_base = next_base;
        }

        accumulate_tile(
            current, x, current_base, physical_wave, lane, partial);
    }

    const std::uint64_t tail_base = full_tiles * kTileK;
    if(logical_group == groups - 1 && physical_wave == 0)
    {
        for(std::uint64_t col = tail_base + static_cast<std::uint64_t>(lane);
            col < n;
            col += kWaveSize)
        {
            partial = fmaf(A[row_base + col], x[col], partial);
        }
    }

    const float wave_sum = reduce_subgroup<32>(partial);
    if(lane == 0)
        wave_partials[physical_wave] = wave_sum;

    __syncthreads();

    if(thread == 0)
    {
        // The main dot-product path stays FP32/FMA. Only the eight wave
        // partials are stitched with compensated FP32 summation. This costs
        // a handful of scalar operations in one thread per workgroup and
        // avoids cancellation leakage at adversarial partition boundaries.
        const float block_sum = compensated_sum8(wave_partials);

        const std::uint64_t slot =
            row * static_cast<std::uint64_t>(groups) +
            static_cast<std::uint64_t>(logical_group);
        block_partials[slot] = block_sum;
        __threadfence();

        const unsigned int ticket = atomicAdd(row_counters + row, 1u);
        if(ticket == static_cast<unsigned int>(groups - 1))
        {
            const std::uint64_t base =
                row * static_cast<std::uint64_t>(groups);

            // Final grid-split stitching is also compensated in FP32.
            // No FP64 accumulation is used: the accuracy improvement comes
            // from a better summation algorithm, not higher-precision math.
            const float total = compensated_sum_dynamic(
                block_partials + base, groups);

            y[row] = total;
            atomicExch(row_counters + row, 0u);
        }
    }
}

enum class SpecializedPath : unsigned char
{
    None,
    Packed4x1,
    Packed4x2,
    Packed8x1,
    Packed8x1Acc2,
    Rows2Cached,
    Rows4Cached,
    Rows8Cached,
    Rows2NonTemporal,
    Rows4NonTemporal,
    Rows8NonTemporal,
    Row4096Static,
    Row4095Persistent,
    Row4097Persistent,
    Split2,
    Split4LDS
};

__host__ __forceinline__ SpecializedPath select_specialized_path(
    std::uint64_t m,
    std::uint64_t n)
{
    if(n == 16 && m >= 262144)
        return SpecializedPath::Packed4x2;

    if(n == 31)
    {
        if(m >= 262144) return SpecializedPath::Packed8x1;
        if(m >= 32768)  return SpecializedPath::Rows8Cached;
    }

    if(n == 32)
    {
        if(m >= 4194304) return SpecializedPath::Rows8NonTemporal;
        if(m >= 524288)  return SpecializedPath::Rows4NonTemporal;
        if(m >= 65536)   return SpecializedPath::Rows4Cached;
        if(m >= 32768)   return SpecializedPath::Rows2NonTemporal;
    }

    if(n == 33)
    {
        if(m >= 262144) return SpecializedPath::Packed8x1;
        if(m >= 32768)  return SpecializedPath::Packed4x1;
    }

    if(n == 63 && m >= 262144)
        return SpecializedPath::Rows2Cached;

    if(n == 64 && m >= 1048576)
        return SpecializedPath::Rows2Cached;

    if(n == 65)
    {
        if(m >= 262144 && m < 2097152)
            return SpecializedPath::Packed4x1;
        if(m >= 32768 && m < 262144)
            return SpecializedPath::Rows2Cached;
    }

    if(n == 129 && m >= 1048576)
        return SpecializedPath::Packed8x1Acc2;

    if((n == 127 || n == 129) && m >= 262144)
        return SpecializedPath::Rows2Cached;

    if(n == 4096 && m == 2049)
        return SpecializedPath::Row4096Static;

    if(n == 4095 && m == 65536)
        return SpecializedPath::Row4095Persistent;

    if(n == 4095 && m >= 65536 && m < 131072)
        return SpecializedPath::Split2;

    if(n == 4097 && m == 65536)
        return SpecializedPath::Row4097Persistent;

    if(n == 4097 && m >= 65536 && m < 131072)
        return SpecializedPath::Split4LDS;

    return SpecializedPath::None;
}

__host__ __forceinline__ bool matrix_at_least(
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t bytes)
{
    const std::uint64_t elements = bytes / sizeof(float);
    return m != 0 && n >= (elements + m - 1) / m;
}

__host__ __forceinline__ bool use_prefetch(
    std::uint64_t m,
    std::uint64_t n)
{
    if(n < kPrefetchMinN)
        return false;
    const std::uint64_t max_elements =
        kPrefetchMaxMatrixBytes / sizeof(float);
    return m != 0 && n <= max_elements / m;
}

__host__ __forceinline__ bool use_split4(
    std::uint64_t m,
    std::uint64_t n)
{
    return m <= kSplitMaxM && n >= kSplitMinN;
}

__host__ __forceinline__ bool use_grid_split(
    std::uint64_t m,
    std::uint64_t n)
{
    return m <= kGridSplitMaxM && n >= kGridSplitMinN;
}

__host__ __forceinline__ bool use_stable_streaming(
    std::uint64_t m,
    std::uint64_t n)
{
    return m >= kStableAccumMinM && n >= kStableAccumMinN;
}

__host__ __forceinline__ bool use_streaming(
    std::uint64_t m,
    std::uint64_t n)
{
    if(n == 4096 && m >= 8191)
        return true;

    return matrix_at_least(m, n, kStreamingMatrixBytes);
}

__host__ __forceinline__ int split_rotation(std::uint64_t n)
{
    return n >= kSplitRotation3MinN ? 3 : 1;
}

__host__ __forceinline__ int grid_chunk_k(
    std::uint64_t m,
    std::uint64_t n)
{
    if(m <= 2)
    {
        if(n < 32768) return 2048;
        if(n < 65536) return 4096;
        return 8192;
    }
    if(m <= 4)
        return 4096;
    if(m <= 16)
        return n < 65536 ? 4096 : 8192;
    if(m <= 32)
    {
        if(n < 32768) return 4096;
        if(n < 65536) return 8192;
        return 16384;
    }
    return n < 32768 ? 8192 : 16384;
}

__host__ __forceinline__ int grid_groups_for(
    std::uint64_t n,
    int chunk_k)
{
    const std::uint64_t full_tiles = n / kTileK;
    const std::uint64_t tiles_per_group =
        static_cast<std::uint64_t>(chunk_k / kTileK);
    return static_cast<int>(
        (full_tiles + tiles_per_group - 1) / tiles_per_group);
}

__host__ __forceinline__ bool valid_grid_x(std::uint64_t blocks)
{
    return blocks != 0 &&
           blocks <= static_cast<std::uint64_t>(
               std::numeric_limits<unsigned int>::max());
}

template<int Width, int Rows>
hipError_t launch_pack(
    const float* A,
    const float* x,
    float* y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    constexpr std::uint64_t rows_per_wave =
        static_cast<std::uint64_t>((kWaveSize / Width) * Rows);
    constexpr std::uint64_t rows_per_block =
        static_cast<std::uint64_t>(kWavesPerBlock) * rows_per_wave;
    const std::uint64_t blocks =
        (m + rows_per_block - 1) / rows_per_block;

    if(!valid_grid_x(blocks))
        return hipErrorInvalidConfiguration;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pack_kernel<Width, Rows>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(kBlockSize), 0, stream,
        A, x, y, m, n, lda);
    return hipGetLastError();
}

template<int Width>
hipError_t launch_pack_acc2(
    const float* A,
    const float* x,
    float* y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    constexpr std::uint64_t rows_per_wave =
        static_cast<std::uint64_t>(kWaveSize / Width);
    constexpr std::uint64_t rows_per_block =
        static_cast<std::uint64_t>(kWavesPerBlock) * rows_per_wave;

    const std::uint64_t blocks =
        (m + rows_per_block - 1) / rows_per_block;

    if(!valid_grid_x(blocks))
        return hipErrorInvalidConfiguration;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pack_acc2_kernel<Width>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(kBlockSize), 0, stream,
        A, x, y, m, n, lda);

    return hipGetLastError();
}

template<int Rows, bool NT>
hipError_t launch_rows(
    const float* A,
    const float* x,
    float* y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    constexpr std::uint64_t rows_per_block =
        static_cast<std::uint64_t>(kWavesPerBlock * Rows);
    const std::uint64_t blocks =
        (m + rows_per_block - 1) / rows_per_block;

    if(!valid_grid_x(blocks))
        return hipErrorInvalidConfiguration;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rows_kernel<Rows, NT>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(kBlockSize), 0, stream,
        A, x, y, m, n, lda);
    return hipGetLastError();
}

hipError_t launch_row4096_static(
    const float* A,
    const float* x,
    float* y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    if(m != 2049u || n != 4096u)
        return hipErrorInvalidValue;

    constexpr unsigned int blocks =
        static_cast<unsigned int>(
            (2049u + kWavesPerBlock - 1u) /
            kWavesPerBlock);

    hipLaunchKernelGGL(
        row4096_static_kernel,
        dim3(blocks),
        dim3(kBlockSize), 0, stream,
        A, x, y, m, lda);

    return hipGetLastError();
}

hipError_t launch_row4095_persistent(
    const float* A,
    const float* x,
    float* y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    if(m != 65536u || n != 4095u)
        return hipErrorInvalidValue;

    hipLaunchKernelGGL(
        row4095_persistent_kernel,
        dim3(kRow4095PersistentBlocks),
        dim3(kBlockSize), 0, stream,
        A, x, y, m, n, lda);
    return hipGetLastError();
}


hipError_t launch_row4097_persistent(
    const float* A,
    const float* x,
    float* y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    if(m != 65536u || n != 4097u)
        return hipErrorInvalidValue;

    hipLaunchKernelGGL(
        row4097_persistent_kernel,
        dim3(kRow4097PersistentBlocks),
        dim3(kBlockSize), 0, stream,
        A, x, y, m, n, lda);
    return hipGetLastError();
}

hipError_t launch_split2(
    const float* A,
    const float* x,
    float* y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    const std::uint64_t blocks = (m + 3u) / 4u;
    if(!valid_grid_x(blocks))
        return hipErrorInvalidConfiguration;

    hipLaunchKernelGGL(
        split2_kernel,
        dim3(static_cast<unsigned int>(blocks)),
        dim3(kBlockSize), 0, stream,
        A, x, y, m, n, lda);
    return hipGetLastError();
}

hipError_t launch_split4_lds(
    const float* A,
    const float* x,
    float* y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    const std::uint64_t blocks = (m + 1u) / 2u;
    if(!valid_grid_x(blocks))
        return hipErrorInvalidConfiguration;

    hipLaunchKernelGGL(
        split4_lds_kernel,
        dim3(static_cast<unsigned int>(blocks)),
        dim3(kBlockSize), 0, stream,
        A, x, y, m, n, lda);
    return hipGetLastError();
}

hipError_t launch_specialized_path(
    SpecializedPath p,
    const float* A,
    const float* x,
    float* y,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    switch(p)
    {
        case SpecializedPath::Packed4x1:
            return launch_pack<4,1>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Packed4x2:
            return launch_pack<4,2>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Packed8x1:
            return launch_pack<8,1>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Packed8x1Acc2:
            return launch_pack_acc2<8>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Rows2Cached:
            return launch_rows<2,false>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Rows4Cached:
            return launch_rows<4,false>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Rows8Cached:
            return launch_rows<8,false>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Rows2NonTemporal:
            return launch_rows<2,true>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Rows4NonTemporal:
            return launch_rows<4,true>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Rows8NonTemporal:
            return launch_rows<8,true>(A,x,y,m,n,lda,stream);
        case SpecializedPath::Row4096Static:
            return launch_row4096_static(A,x,y,m,n,lda,stream);
        case SpecializedPath::Row4095Persistent:
            return launch_row4095_persistent(A,x,y,m,n,lda,stream);
        case SpecializedPath::Row4097Persistent:
            return launch_row4097_persistent(A,x,y,m,n,lda,stream);
        case SpecializedPath::Split2:
            return launch_split2(A,x,y,m,n,lda,stream);
        case SpecializedPath::Split4LDS:
            return launch_split4_lds(A,x,y,m,n,lda,stream);
        default:
            return hipErrorInvalidValue;
    }
}

template<int Width>
hipError_t launch_small(const float* A,
                        const float* x,
                        float* y,
                        std::uint64_t m,
                        std::uint64_t n,
                        std::uint64_t lda,
                        hipStream_t stream)
{
    constexpr std::uint64_t outputs_per_wave =
        static_cast<std::uint64_t>(kWaveSize / Width);
    constexpr std::uint64_t outputs_per_block =
        static_cast<std::uint64_t>(kWavesPerBlock) * outputs_per_wave;
    const std::uint64_t blocks = (m + outputs_per_block - 1) / outputs_per_block;
    if(!valid_grid_x(blocks))
        return hipErrorInvalidConfiguration;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(small_n_kernel<Width>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(kBlockSize), 0, stream,
        A, x, y, m, n, lda);
    return hipGetLastError();
}

template<bool NT, bool Acc4>
hipError_t launch_row(const float* A,
                      const float* x,
                      float* y,
                      std::uint64_t m,
                      std::uint64_t n,
                      std::uint64_t lda,
                      hipStream_t stream)
{
    const std::uint64_t blocks =
        (m + static_cast<std::uint64_t>(kWavesPerBlock) - 1) /
        static_cast<std::uint64_t>(kWavesPerBlock);
    if(!valid_grid_x(blocks))
        return hipErrorInvalidConfiguration;

    if constexpr(!Acc4 && !NT)
    {
        if(use_prefetch(m, n))
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(row_kernel<NT, Acc4, true>),
                dim3(static_cast<unsigned int>(blocks)), dim3(kBlockSize),
                0, stream, A, x, y, m, n, lda);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(row_kernel<NT, Acc4, false>),
                dim3(static_cast<unsigned int>(blocks)), dim3(kBlockSize),
                0, stream, A, x, y, m, n, lda);
        }
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(row_kernel<NT, Acc4, false>),
            dim3(static_cast<unsigned int>(blocks)), dim3(kBlockSize),
            0, stream, A, x, y, m, n, lda);
    }
    return hipGetLastError();
}

hipError_t launch_split4(const float* A,
                         const float* x,
                         float* y,
                         std::uint64_t m,
                         std::uint64_t n,
                         std::uint64_t lda,
                         hipStream_t stream)
{
    constexpr std::uint64_t outputs_per_block = 2;
    const std::uint64_t blocks = (m + outputs_per_block - 1) / outputs_per_block;
    if(!valid_grid_x(blocks))
        return hipErrorInvalidConfiguration;

    if(split_rotation(n) == 1)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(split4_kernel<1>),
            dim3(static_cast<unsigned int>(blocks)), dim3(kBlockSize),
            0, stream, A, x, y, m, n, lda);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(split4_kernel<3>),
            dim3(static_cast<unsigned int>(blocks)), dim3(kBlockSize),
            0, stream, A, x, y, m, n, lda);
    }
    return hipGetLastError();
}

template<int TilesPerGroup>
hipError_t launch_grid_split_impl(
    const float* A,
    const float* x,
    float* y,
    float* block_partials,
    unsigned int* row_counters,
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t lda,
    hipStream_t stream)
{
    constexpr int chunk_k = TilesPerGroup * kTileK;
    const int groups = grid_groups_for(n, chunk_k);
    const std::uint64_t blocks =
        m * static_cast<std::uint64_t>(groups);
    if(!valid_grid_x(blocks))
        return hipErrorInvalidConfiguration;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(grid_split_kernel<TilesPerGroup>),
        dim3(static_cast<unsigned int>(blocks)), dim3(kBlockSize),
        0, stream,
        A, x, y, block_partials, row_counters, m, n, lda, groups);
    return hipGetLastError();
}

hipError_t launch_grid_split(const float* A,
                             const float* x,
                             float* y,
                             float* block_partials,
                             unsigned int* row_counters,
                             std::uint64_t m,
                             std::uint64_t n,
                             std::uint64_t lda,
                             hipStream_t stream)
{
    switch(grid_chunk_k(m, n))
    {
        case 2048: return launch_grid_split_impl<8>(A,x,y,block_partials,row_counters,m,n,lda,stream);
        case 4096: return launch_grid_split_impl<16>(A,x,y,block_partials,row_counters,m,n,lda,stream);
        case 8192: return launch_grid_split_impl<32>(A,x,y,block_partials,row_counters,m,n,lda,stream);
        default:   return launch_grid_split_impl<64>(A,x,y,block_partials,row_counters,m,n,lda,stream);
    }
}

hipError_t launch_large(const float* A,
                        const float* x,
                        float* y,
                        float* block_partials,
                        unsigned int* row_counters,
                        std::uint64_t m,
                        std::uint64_t n,
                        std::uint64_t lda,
                        hipStream_t stream)
{
    if(use_grid_split(m, n))
    {
        if(block_partials == nullptr || row_counters == nullptr)
            return hipErrorInvalidValue;
        return launch_grid_split(
            A, x, y, block_partials, row_counters, m, n, lda, stream);
    }

    if(use_split4(m, n))
        return launch_split4(A, x, y, m, n, lda, stream);

    if(use_stable_streaming(m, n))
        return launch_row<true, true>(
            A, x, y, m, n, lda, stream);

    if(use_streaming(m, n))
        return launch_row<true, false>(
            A, x, y, m, n, lda, stream);

    return launch_row<false, false>(
        A, x, y, m, n, lda, stream);
}

const char* implementation_name(std::uint64_t m, std::uint64_t n)
{
    switch(select_specialized_path(m,n))
    {
        case SpecializedPath::Packed4x1:
        case SpecializedPath::Packed4x2:
        case SpecializedPath::Packed8x1:
        case SpecializedPath::Packed8x1Acc2:
            return "PACK";
        case SpecializedPath::Rows2Cached:
        case SpecializedPath::Rows4Cached:
        case SpecializedPath::Rows8Cached:
        case SpecializedPath::Rows2NonTemporal:
        case SpecializedPath::Rows4NonTemporal:
        case SpecializedPath::Rows8NonTemporal:
            return "ROWS";
        case SpecializedPath::Row4096Static:
            return "ROW_STATIC";
        case SpecializedPath::Row4095Persistent:
        case SpecializedPath::Row4097Persistent:
            return "PERSISTENT_REG";
        case SpecializedPath::Split2:
            return "K_SPLIT2";
        case SpecializedPath::Split4LDS:
            return "K_SPLIT4_LDS";
        default:
            break;
    }

    if(n <= 16) return "SMALL_N";
    if(use_grid_split(m, n)) return "GRID_SPLIT";
    if(use_split4(m, n)) return "SPLIT4";
    if(use_stable_streaming(m, n)) return "STREAMING_ACC4";
    if(use_streaming(m, n)) return "STREAMING";
    return use_prefetch(m, n) ? "ROW_PREFETCH" : "ROW";
}

}

static std::uint64_t partial_bytes(std::uint64_t m,std::uint64_t n)
{
    if(!use_grid_split(m,n)) return 0;
    return m*static_cast<std::uint64_t>(
        grid_groups_for(n,grid_chunk_k(m,n)))*sizeof(float);
}

extern "C" hipError_t gemv(
    const float* A,const float* x,float* y,
    std::int64_t m,std::int64_t n,std::int64_t lda,
    void* workspace,hipStream_t stream)
{
    if(!A||!x||!y||m<=0||n<=0||lda<n) return hipErrorInvalidValue;
    const auto M=static_cast<std::uint64_t>(m);
    const auto N=static_cast<std::uint64_t>(n);
    const auto L=static_cast<std::uint64_t>(lda);

    const SpecializedPath tp = select_specialized_path(M,N);
    if(tp != SpecializedPath::None)
        return launch_specialized_path(tp,A,x,y,M,N,L,stream);

    if(N<=1) return launch_small<1>(A,x,y,M,N,L,stream);
    if(N<=2) return launch_small<2>(A,x,y,M,N,L,stream);
    if(N<=4) return launch_small<4>(A,x,y,M,N,L,stream);
    if(N<=8) return launch_small<8>(A,x,y,M,N,L,stream);
    if(N<=16)return launch_small<16>(A,x,y,M,N,L,stream);
    float* p=nullptr;
    unsigned int* c=nullptr;
    if(use_grid_split(M,N)){
        if(!workspace) return hipErrorInvalidValue;
        const auto b=partial_bytes(M,N);
        p=static_cast<float*>(workspace);
        c=reinterpret_cast<unsigned int*>(
            static_cast<unsigned char*>(workspace)+b);
    }
    return launch_large(A,x,y,p,c,M,N,L,stream);
}

extern "C" std::uint64_t gemv_workspace_bytes(
    std::int64_t m,std::int64_t n)
{
    if(m<=0||n<=0) return 0;
    const auto M=static_cast<std::uint64_t>(m);
    const auto N=static_cast<std::uint64_t>(n);
    if(!use_grid_split(M,N)) return 0;
    return partial_bytes(M,N)+M*sizeof(unsigned int);
}

extern "C" const char* gemv_path_name(std::int64_t m,std::int64_t n)
{
    if(m<=0||n<=0) return "INVALID";
    return implementation_name(
        static_cast<std::uint64_t>(m),
        static_cast<std::uint64_t>(n));
}
