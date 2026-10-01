#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

#include "filter_impl.h"
#include "logo.h"

#define K 10
#define RGB_DIFF_THRESHOLD 10
constexpr uint8_t MAX_WEIGHTS = 60;

#define ONE_THIRD (1.0f / 3.0f)
#define HYSTERESIS_LOW 35
#define HYSTERESIS_HIGH 60

#define TILE_X 16
#define TILE_Y 16

struct rgb_uncapped
{
    int r, g, b;
};

struct rgb
{
    uint8_t r, g, b;
};

struct rgbw
{
    rgb RGB;
    uint8_t w;
};

typedef rgbw pool[K];

__constant__ uint8_t* logo;

__device__ inline int get_pool_offset(int x, int y, int width)
{
    return y * width + x;
}

__device__ inline uint8_t clamp_u8(int v)
{
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

__device__ inline const rgb& pitched_pixel(const uint8_t* pitched, size_t pitch,
                                           int x, int y)
{
    const rgb* lineptr = (const rgb*)(pitched + y * pitch);
    return lineptr[x];
}

__device__ inline void write_pitched_pixel(uint8_t* pitched, size_t pitch,
                                           int x, int y, const rgb& v)
{
    rgb* lineptr = (rgb*)(pitched + y * pitch);
    lineptr[x] = v;
}

__device__ uint32_t hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__ uint32_t random_pixel(uint32_t x, uint32_t y, uint32_t seed)
{
    uint32_t h = seed;
    h ^= hash(x);
    h ^= hash(y + 0x9e3779b9);
    return hash(h);
}

__device__ float random_float(uint32_t x, uint32_t y, uint32_t seed)
{
    return random_pixel(x, y, seed) * (1.0f / 4294967296.0f);
}

__device__ rgb_uncapped mult_rgb_w(rgb p, uint8_t w)
{
    return rgb_uncapped{ p.r * w, p.g * w, p.b * w };
}

__device__ rgb_uncapped add_rgbs(rgb_uncapped one, rgb other)
{
    one.r += other.r;
    one.g += other.g;
    one.b += other.b;
    return one;
}

__device__ rgb div_rgb_w(rgb_uncapped p, uint8_t w)
{
    p.r /= w;
    p.g /= w;
    p.b /= w;
    return rgb{ static_cast<uint8_t>(p.r), static_cast<uint8_t>(p.g),
                static_cast<uint8_t>(p.b) };
}

__device__ rgb min_rgb(rgb b, rgb m)
{
    return rgb{ static_cast<uint8_t>(abs(int(b.r) - int(m.r))),
                static_cast<uint8_t>(abs(int(b.g) - int(m.g))),
                static_cast<uint8_t>(abs(int(b.b) - int(m.b))) };
}

__device__ rgb apply_mask(rgb background, bool mask)
{
    if (mask)
        return rgb{ clamp_u8(int(background.r) + 127), background.g,
                    background.b };
    return background;
}

__global__ void reservoir_filter_kernel(const uint8_t* orig_pitched,
                                        size_t orig_pitch, rgb* estimate,
                                        int width, int height, pool* pools,
                                        unsigned int* rands)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int pool_offset = get_pool_offset(x, y, width);
    rgb p = pitched_pixel(orig_pitched, orig_pitch, x, y);
    pool& rs = pools[pool_offset];

    int match_idx = -1;
    int first_empty = -1;
    int min_idx = 0, max_idx = 0;
    uint8_t min_val = 255, max_val = 0;
    int total = 0;

    for (int i = 0; i < K; ++i)
    {
        uint8_t w = rs[i].w;
        if (w > 0)
        {
            if (match_idx == -1
                && abs(int(p.r) - int(rs[i].RGB.r)) < RGB_DIFF_THRESHOLD
                && abs(int(p.g) - int(rs[i].RGB.g)) < RGB_DIFF_THRESHOLD
                && abs(int(p.b) - int(rs[i].RGB.b)) < RGB_DIFF_THRESHOLD)
            {
                match_idx = i;
            }
        }
        else if (first_empty == -1)
        {
            first_empty = i;
        }

        if (w < min_val)
        {
            min_val = w;
            min_idx = i;
        }
        if (w > max_val)
        {
            max_val = w;
            max_idx = i;
        }
        total += w;
    }

    int m_idx = (match_idx != -1) ? match_idx : first_empty;

    if (m_idx != -1 && rs[m_idx].w > 0)
    {
        uint8_t neww = rs[m_idx].w + 1;
        rs[m_idx].RGB = div_rgb_w(
            add_rgbs(mult_rgb_w(rs[m_idx].RGB, rs[m_idx].w), p), neww);
        rs[m_idx].w = neww;
        if (neww > max_val)
            max_idx = m_idx;
    }
    else if (m_idx != -1 && rs[m_idx].w == 0)
    {
        rs[m_idx].RGB = p;
        rs[m_idx].w = 1;
        if (1 > max_val)
            max_idx = m_idx;
    }
    else
    {
        if (random_float(x, y, rands[pool_offset]) * total >= min_val)
        {
            rs[min_idx].RGB = p;
            rs[min_idx].w = 1;
            if (min_idx == max_idx || 1 > max_val)
                max_idx = (1 >= max_val) ? min_idx : max_idx;
        }
    }

    for (int i = 0; i < K; ++i)
    {
        if (rs[i].w > MAX_WEIGHTS)
            rs[i].w = MAX_WEIGHTS;
    }

    uint8_t best_w = rs[0].w;
    int best_idx = 0;
    for (int i = 1; i < K; ++i)
    {
        if (rs[i].w > best_w)
        {
            best_w = rs[i].w;
            best_idx = i;
        }
    }

    estimate[pool_offset] = rs[best_idx].RGB;
}

__global__ void erode_kernel(const rgb* in,
                             rgb* out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uint8_t r = 255, g = 255, b = 255;
    for (int dy = -1; dy <= 1; ++dy)
    {
        int cy = min(max(y + dy, 0), height - 1);
        for (int dx = -1; dx <= 1; ++dx)
        {
            int cx = min(max(x + dx, 0), width - 1);
            rgb p = in[cy * width + cx];
            r = min(r, p.r);
            g = min(g, p.g);
            b = min(b, p.b);
        }
    }
    out[y * width + x] = rgb{ r, g, b };
}

__global__ void dilate_kernel(const rgb* in,
                              rgb* out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uint8_t r = 0, g = 0, b = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        int cy = min(max(y + dy, 0), height - 1);
        for (int dx = -1; dx <= 1; ++dx)
        {
            int cx = min(max(x + dx, 0), width - 1);
            rgb p = in[cy * width + cx];
            r = max(r, p.r);
            g = max(g, p.g);
            b = max(b, p.b);
        }
    }
    out[y * width + x] = rgb{ r, g, b };
}

__global__ void diff_threshold_kernel(const uint8_t* orig_pitched,
                                      size_t orig_pitch, const rgb* opened,
                                      bool* min_img, bool* max_img, bool* res,
                                      int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int i = y * width + x;
    rgb d = min_rgb(pitched_pixel(orig_pitched, orig_pitch, x, y), opened[i]);
    float avg = d.r * ONE_THIRD + d.g * ONE_THIRD + d.b * ONE_THIRD;

    bool lo = avg > HYSTERESIS_LOW;
    bool hi = avg > HYSTERESIS_HIGH;
    min_img[i] = lo;
    max_img[i] = hi;
    res[i] = hi;
}

__global__ void hysteresis_propagate_kernel(bool* res, const bool* min_img,
                                            int width, int height, int* changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int i = y * width + x;

    if (res[i] || !min_img[i])
        return;

    bool neighbour_selected = (x > 0 && res[i - 1])
        || (x < width - 1 && res[i + 1]) || (y > 0 && res[i - width])
        || (y < height - 1 && res[i + width]);

    if (neighbour_selected)
    {
        res[i] = true;
        atomicOr(changed, 1);
    }
}

__global__ void composite_kernel(const uint8_t* orig_pitched, size_t orig_pitch,
                                 const bool* mask, uint8_t* out_pitched,
                                 size_t out_pitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int i = y * width + x;
    rgb result =
        apply_mask(pitched_pixel(orig_pitched, orig_pitch, x, y), mask[i]);
    write_pitched_pixel(out_pitched, out_pitch, x, y, result);
}

__global__ void init_pools_kernel(pool* pools, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    for (int k = 0; k < K; ++k)
    {
        pools[i][k].RGB = rgb{ 0, 0, 0 };
        pools[i][k].w = 0;
    }
}

__global__ void init_rands_kernel(unsigned int* rands, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    rands[i] = 0x9e3779b9u ^ static_cast<unsigned int>(i);
}

extern "C"
{
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride,
                     int pixel_stride)
    {
        assert(sizeof(rgb) == pixel_stride);

        const int n = width * height;
        dim3 blockSize(TILE_X, TILE_Y);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);

        static pool* d_pools = nullptr;
        static unsigned int* d_rands = nullptr;
        static bool initialized = false;

        static rgb *d_estimate = nullptr, *d_eroded = nullptr,
                   *d_opened = nullptr;
        static bool *d_min_img = nullptr, *d_max_img = nullptr,
                    *d_hysteresis = nullptr;
        static std::byte* dPitched = nullptr;
        static size_t pitch = 0;
        static int* d_changed = nullptr;
        static int* h_changed_pinned = nullptr;
        static cudaStream_t stream;

        if (!initialized)
        {
            cudaMalloc(&d_pools, n * sizeof(pool));
            cudaMalloc(&d_rands, n * sizeof(unsigned int));

            int initBlock = 256;
            int initGrid = (n + initBlock - 1) / initBlock;
            init_pools_kernel<<<initGrid, initBlock>>>(d_pools, n);
            init_rands_kernel<<<initGrid, initBlock>>>(d_rands, n);

            pitch = width * sizeof(rgb);
            cudaMalloc(&dPitched, pitch * height);

            cudaMalloc(&d_estimate, n * sizeof(rgb));
            cudaMalloc(&d_eroded, n * sizeof(rgb));
            cudaMalloc(&d_opened, n * sizeof(rgb));
            cudaMalloc(&d_min_img, n * sizeof(bool));
            cudaMalloc(&d_max_img, n * sizeof(bool));
            cudaMalloc(&d_hysteresis, n * sizeof(bool));
            cudaMalloc(&d_changed, sizeof(int));
            cudaHostAlloc(&h_changed_pinned, sizeof(int), cudaHostAllocDefault);
            cudaStreamCreate(&stream);

            initialized = true;
        }

        cudaMemcpy2DAsync(dPitched, pitch, src_buffer, src_stride,
                          width * sizeof(rgb), height, cudaMemcpyDefault,
                          stream);

        reservoir_filter_kernel<<<gridSize, blockSize, 0, stream>>>(
            (uint8_t*)dPitched, pitch, d_estimate, width, height, d_pools,
            d_rands);

        erode_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_estimate, d_eroded, width, height);

        dilate_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_eroded, d_opened, width, height);

        diff_threshold_kernel<<<gridSize, blockSize, 0, stream>>>(
            (uint8_t*)dPitched, pitch, d_opened, d_min_img, d_max_img,
            d_hysteresis, width, height);

        int h_changed;
        do
        {
            cudaMemsetAsync(d_changed, 0, sizeof(int), stream);
            hysteresis_propagate_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_hysteresis, d_min_img, width, height, d_changed);

            cudaMemcpyAsync(h_changed_pinned, d_changed, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            h_changed = *h_changed_pinned;
        } while (h_changed != 0);

        composite_kernel<<<gridSize, blockSize, 0, stream>>>(
            (uint8_t*)dPitched, pitch, d_hysteresis, (uint8_t*)dPitched, pitch,
            width, height);

        cudaMemcpy2DAsync(src_buffer, src_stride, dPitched, pitch,
                          width * sizeof(rgb), height, cudaMemcpyDefault,
                          stream);

        cudaStreamSynchronize(stream);


        {
            using namespace std::chrono_literals;
        }
    }
}
