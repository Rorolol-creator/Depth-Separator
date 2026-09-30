#include "filter_impl.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>

#include "logo.h"

#define K 32
#define RGB_DIFF_THRESHOLD 50
constexpr uint8_t MAX_WEIGHTS = 32;

#define ONE_THIRD (1.0/3.0)
#define HYSTERESIS_LOW 10
#define HYSTERESIS_HIGH 50

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

typedef std::array<rgbw, K> pool;

extern "C"
{
    uint32_t hash(uint32_t x)
    {
        x ^= x >> 16;
        x *= 0x7feb352d;
        x ^= x >> 15;
        x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    }

    uint32_t random_pixel(uint32_t x, uint32_t y, uint32_t seed)
    {
        uint32_t h = seed;
        h ^= hash(x);
        h ^= hash(y + 0x9e3779b9);
        return hash(h);
    }

    float random_float(uint32_t x, uint32_t y, uint32_t seed)
    {
        return random_pixel(x, y, seed) * (1.0f / 4294967296.0f);
    }

    void print_pool(const std::vector<pool>& rs, int offset)
    {
        for (int i = 0; i < K; ++i)
        {
            if (rs[offset][i].w != 0)
                std::cout << static_cast<int>(rs[offset][i].w) << ", "
                          << std::endl;
        }
    }
    int find_matching_reservoir(const rgbw& p, const pool& pool)
    {
        int m_idx = -1;
        for (int i = 0; i < K; ++i)
        {
            if (pool[i].w > 0)
            {
                if (std::abs(static_cast<int>(p.RGB.r)
                             - static_cast<int>(pool[i].RGB.r))
                        < RGB_DIFF_THRESHOLD
                    && std::abs(static_cast<int>(p.RGB.g)
                                - static_cast<int>(pool[i].RGB.g))
                        < RGB_DIFF_THRESHOLD
                    && std::abs(static_cast<int>(p.RGB.b)
                                - static_cast<int>(pool[i].RGB.b))
                        < RGB_DIFF_THRESHOLD)
                {
                    return i;
                }
            }
            else
            {
                m_idx = i;
            }
        }
        return m_idx;
    }

    int find_min_reservoir(const std::vector<pool>& rs, int offset)
    {
        uint8_t min_val = rs[offset][0].w;
        int min_idx = 0;
        for (int i = 0; i < K; ++i)
        {
            if (rs[offset][i].w < min_val)
            {
                min_val = rs[offset][i].w;
                min_idx = i;
            }
        }
        return min_idx;
    }

    int find_max_reservoir(const std::vector<pool>& rs, int offset)
    {
        uint8_t max_val = rs[offset][0].w;
        int max_idx = 0;
        for (int i = 0; i < K; ++i)
        {
            if (rs[offset][i].w > max_val)
            {
                max_val = rs[offset][i].w;
                max_idx = i;
            }
        }
        return max_idx;
    }

    int sum_reservoirs(const std::vector<pool>& rs, int offset)
    {
        int sum = rs[offset][0].w;
        for (int i = 1; i < K; ++i)
        {
            sum += rs[offset][i].w;
        }
        return sum;
    }

    void cap_reservoirs(std::vector<pool>& rs, int offset)
    {
        for (int i = 0; i < K; ++i)
        {
            rs[offset][i].w = std::min(rs[offset][i].w, MAX_WEIGHTS);
        }
    }

    static std::vector<pool> init_pool(int w, int h)
    {
        static std::vector<pool> rs = std::vector<pool>();
        rs.resize(K * w * h);

        for (int i = 0; i < w * h; ++i)
        {
            for (int j = 0; j < K; ++j)
            {
                rs[i][j].w = 0;
            }
        }

        return rs;
    }

    static std::vector<int> init_rand(int w, int h)
    {
        static std::vector<int> rands = std::vector<int>();
        rands.resize(w * h);

        for (unsigned i = 0; i < w * h; ++i)
        {
            rands[i] = i;
        }

        return rands;
    }

    rgb_uncapped mult_rgb_w(rgb p, uint8_t w)
    {
        return rgb_uncapped{ p.r * w, p.g * w, p.b * w };
    }

    rgb min_rgb(rgb b, rgb m)
    {
        return rgb{ static_cast<uint8_t>(std::abs(b.r - m.r)),
                    static_cast<uint8_t>(std::abs(b.g - m.g)),
                    static_cast<uint8_t>(std::abs(b.b - m.b)) };
    }

    rgb div_rgb_w(rgb_uncapped p, uint8_t w)
    {
        p.r /= w;
        p.g /= w;
        p.b /= w;
        return rgb{ static_cast<uint8_t>(p.r), static_cast<uint8_t>(p.g),
                    static_cast<uint8_t>(p.b) };
    }

    rgb_uncapped add_rgbs(rgb_uncapped one, rgb other)
    {
        one.r += other.r;
        one.g += other.g;
        one.b += other.b;
        return one;
    }

    int get_pool_offset(int x, int y, int w)
    {
        return x + y * w;
    }

    void opening(uint8_t* buffer, int width, int height, int stride)
    {
        std::vector<rgb> eroded(width * height);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                uint8_t min_r = 255, min_g = 255, min_b = 255;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int nx = std::clamp(x + dx, 0, width - 1);
                        int ny = std::clamp(y + dy, 0, height - 1);
                        const rgb& p = ((rgb*)(buffer + ny * stride))[nx];
                        min_r = std::min(min_r, p.r);
                        min_g = std::min(min_g, p.g);
                        min_b = std::min(min_b, p.b);
                    }
                eroded[y * width + x] = { min_r, min_g, min_b };
            }
        }

        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*)(buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                uint8_t max_r = 0, max_g = 0, max_b = 0;
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int nx = std::clamp(x + dx, 0, width - 1);
                        int ny = std::clamp(y + dy, 0, height - 1);
                        const rgb& p = eroded[ny * width + nx];
                        max_r = std::max(max_r, p.r);
                        max_g = std::max(max_g, p.g);
                        max_b = std::max(max_b, p.b);
                    }
                }
                lineptr[x] = { max_r, max_g, max_b };
            }
        }
    }

    uint8_t* copy(uint8_t* buffer, int width, int height, int stride)
    {
        uint8_t* buf = new uint8_t[stride * height];
        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*)(buffer + y * stride);
            rgb* new_lineptr = (rgb*)(buf + y * stride);
            for (int x = 0; x < width; ++x)
            {
                new_lineptr[x] = lineptr[x];
            }
        }
        return buf;
    }

    rgb apply_mask(rgb background, bool mask)
    {
        if (mask)
            return rgb{ static_cast<uint8_t>(std::min(
                            static_cast<int>(background.r + 127), 255)),
                        background.g, background.b };
        return background;
    }

    void spread_seen(std::vector<bool>& seen, std::vector<bool>& min_img, int i, int j, int width, int height)
    {
        std::queue<std::pair<int, int>> q{ };
        q.push(std::pair<int, int>(i, j));

        while (!q.empty())
        {
            std::pair<int, int> p = q.front();
            q.pop();

            int pixel = p.second * width + p.first;

            if (!min_img[pixel] || seen[pixel])
                continue;

            seen[pixel] = true;

            if (p.first != 0)
                q.push(std::pair<int, int>(p.first - 1, p.second));
            if (p.first != width - 1)
                q.push(std::pair<int, int>(p.first + 1, p.second));

            if (p.second != 0)
                q.push(std::pair<int, int>(p.first, p.second - 1));
            if (p.second != height - 1)
                q.push(std::pair<int, int>(p.first, p.second + 1));
        }
    }

    std::vector<bool> apply_hysteresis(uint8_t* img, int width, int height)
    {
        rgb* img_rgb = (rgb*)img;

        std::vector<bool> res(width * height);
        std::vector<bool> min_img(width * height);
        std::vector<bool> max_img(width * height);

        for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++)
        {
            auto pixel = img_rgb[(j * width + i)];
            auto avg = pixel.r * ONE_THIRD + pixel.g * ONE_THIRD + pixel.b * ONE_THIRD;

            min_img[j * width + i] = avg > HYSTERESIS_LOW;
            max_img[j * width + i] = avg > HYSTERESIS_HIGH;
        }

        for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++)
        {
            int index = j * width + i;

            if (!max_img[index] || res[index])
                continue;

            spread_seen(res, min_img, i, j, width, height);
        }

        return res;
    }

    void filter_impl(uint8_t* buffer, int width, int height, int stride,
                     int pixel_stride)
    {
        static std::vector<pool> rs = init_pool(width, height);
        static std::vector<int> rands = init_rand(width, height);
        uint8_t* background = copy(buffer, width, height, stride);

        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*)(buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                rgbw p = rgbw{ lineptr[x], 1 };
                int pool_offset = get_pool_offset(x, y, width);

                int m_idx = find_matching_reservoir(p, rs[pool_offset]);

                if (m_idx != -1 && rs[pool_offset][m_idx].w > 0)
                {
                    rs[pool_offset][m_idx].w++;
                    rs[pool_offset][m_idx].RGB = div_rgb_w(
                        add_rgbs(mult_rgb_w(rs[pool_offset][m_idx].RGB,
                                            rs[pool_offset][m_idx].w - 1),
                                 p.RGB),
                        rs[pool_offset][m_idx].w);
                    // print_pool(rs, pool_offset);
                }
                else if (m_idx != -1 && rs[pool_offset][m_idx].w == 0)
                {
                    rs[pool_offset][m_idx].RGB = p.RGB;
                    rs[pool_offset][m_idx].w = 1;
                }
                else
                {
                    int min_idx = find_min_reservoir(rs, pool_offset);
                    int total_weights = sum_reservoirs(rs, pool_offset);
                    if (random_float(x, y, rands[pool_offset]) * total_weights
                        >= rs[pool_offset][min_idx].w)
                    {
                        rs[pool_offset][min_idx].RGB = p.RGB; // TODO check that
                        rs[pool_offset][min_idx].w = 1;
                    }
                }

                cap_reservoirs(rs, pool_offset); // TODO check
                lineptr[x] =
                    rs[pool_offset][find_max_reservoir(rs, pool_offset)].RGB;
            }
        }

        opening(buffer, width, height, stride);

        for (int y = 0; y < height; ++y)
        {
            rgb* mask = (rgb*)(buffer + y * stride);
            rgb* back = (rgb*)(background + y * stride);
            for (int x = 0; x < width; ++x)
            {
                mask[x] = min_rgb(back[x], mask[x]);
            }
        }

        std::vector<bool> hysteresis_img = apply_hysteresis(buffer, width, height);

        for (int y = 0; y < height; ++y)
        {
            rgb* mask = (rgb*)(buffer + y * stride);
            rgb* back = (rgb*)(background + y * stride);
            for (int x = 0; x < width; ++x)
            {
                mask[x] = apply_mask(back[x], hysteresis_img[get_pool_offset(x, y, width)]);
            }
        }

        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            // std::this_thread::sleep_for(100ms);
        }
    }
}
