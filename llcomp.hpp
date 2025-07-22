#pragma once
#include <functional>
#include <cassert>
#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <string_view>
#include <cstdint>
#include <functional>
#include <utility>
#include <type_traits>
#include <future>
#include "bitstream.hpp"
#include "rlgr.hpp"

namespace llcomp
{
    constexpr inline auto ext = ".llr";
    constexpr inline uint8_t rev = '0';
    constexpr inline uint32_t signature = 'l' | 'l' << 8U | 'r' << 16U | rev << 24U;

    inline int median(int a, int b, int c)
    {
        if (a > b)
        {
            if (c > b)
            {
                if (c > a)
                    b = a;
                else
                    b = c;
            }
        }
        else
        {
            if (b > c)
            {
                if (c > a)
                    b = c;
                else
                    b = a;
            }
        }
        return b;
    }

    auto split_8bit_pixels_into_int16_channels(const uint8_t *_data, int channels, int width, int height, size_t stride = 0)
    {
        if (stride == 0)
        {
            stride = width * channels;
        }
        auto res = std::vector(channels, std::vector(width * height, int16_t{}));
        size_t pos{0};

        for (int h = 0; h < height; ++h)
        {
            auto data = _data;
            _data += stride;
            for (int w = 0; w < width; ++w)
            {
                if (channels >= 3)
                {
                    int r = data[0];
                    int g = data[1];
                    int b = data[2];
                    b -= g;
                    r -= g;
                    g += (b + r) / 4;
                    res[0][pos] = static_cast<int16_t>(r);
                    res[1][pos] = static_cast<int16_t>(g);
                    res[2][pos] = static_cast<int16_t>(b);

                    for (int i = 3; i < channels; i++)
                    {
                        res[i][pos] = static_cast<int16_t>(data[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < channels; i++)
                    {
                        res[i][pos] = static_cast<int16_t>(data[i]);
                    }
                }
                data += channels;
                pos += 1;
            }
        }
        return res;
    }

    std::vector<uint8_t> join_int16_channels_to_8bit_pixels(std::vector<std::vector<int16_t>> &channels, uint32_t width, uint32_t height, size_t stride = 0)
    {

        if (stride == 0)
        {
            stride = width * channels.size();
        }
        auto res = std::vector(stride * height, uint8_t{});
        auto out_ = res.data();
        size_t pos = 0;
        for (uint32_t h = 0; h < height; ++h)
        {
            auto out = out_;
            out_ += stride;
            for (uint32_t w = 0; w < width; ++w)
            {
                if (channels.size() >= 3)
                {
                    int r = channels[0][pos];
                    int g = channels[1][pos];
                    int b = channels[2][pos];
                    g -= ((r + b) / 4);
                    r += g;
                    b += g;
                    out[0] = std::max(0, std::min(255, r));
                    out[1] = std::max(0, std::min(255, g));
                    out[2] = std::max(0, std::min(255, b));
                }
                for (size_t n = 3; n < channels.size(); ++n)
                {
                    out[n] = std::max(int16_t{0}, std::min(int16_t{255}, channels[n][pos]));
                }
                out += channels.size();
                pos++;
            }
        }
        return res;
    }

    inline std::vector<uint8_t> encodeChannel(const int16_t *data, uint32_t width, uint32_t height)
    {
        auto size = size_t(width) * height;
        std::vector<uint8_t> buffer;
        buffer.reserve(size * 3 / 4);
        BitStream::Writer bit_stream{[&](const uint8_t *data, size_t size)
                                     {
                                         buffer.resize(buffer.size() + size);
                                         if (size == 8)
                                             std::memcpy(buffer.data() + buffer.size() - 8, data, 8);
                                         else
                                             std::memcpy(buffer.data() + buffer.size() - size, data, size);
                                     }};
        RLGR::Encoder encoder{[&](uint32_t n, uint32_t val)
                              { bit_stream.put_bits(n, val); }};
        for (uint32_t h = 0; h < height; ++h)
        {
            for (uint32_t w = 0; w < width; ++w)
            {
                const int l = w > 0 ? data[-1] : h > 0 ? data[-ptrdiff_t(width)]
                                                       : 0;
                const int t = h > 0 ? data[-ptrdiff_t(width)] : l;
                const int tl = h > 0 && w > 0 ? data[-ptrdiff_t(width) - 1] : t;
                const int predict = median(l, l + t - tl, t);
                int diff = (static_cast<int>(data[0]) - predict);
                encoder.encode(diff);
                data++;
            }
        }
        encoder.encode(-1);
        bit_stream.flush();
        return buffer;
    }

    struct RawImage8bit
    {
        std::vector<uint8_t> pixels;
        uint32_t width;
        uint32_t height;
        uint32_t channel_nb;
    };

    inline void decodeChannel(uint32_t width, uint32_t height, const uint8_t *data_begin, const uint8_t *data_end, int16_t *pixels)
    {
        size_t rgb_pos = 0;
        auto data = data_begin;

        BitStream::Reader bit_stream{[&]()
                                     {
                                         if (data + 8 <= data_end)
                                         {
                                             data += 8;
                                             return std::make_pair(data - 8, size_t{8});
                                         }
                                         else
                                         {
                                             size_t rem = data_end - data;
                                             return std::make_pair(data_end - rem, rem);
                                         }
                                     }};
        bit_stream.init();

        RLGR::Decoder decoder{[&]
                              { return bit_stream.peek32(); }, [&](uint32_t n)
                              { return bit_stream.skip(n); }};

        auto [next, ziros] = decoder.decode();
        for (uint32_t h = 0; h < height; ++h)
        {
            for (uint32_t w = 0; w < width; ++w)
            {

                const int l = w > 0 ? pixels[-1] : (h > 0 ? pixels[-static_cast<ptrdiff_t>(width)] : 0);
                const int t = h > 0 ? pixels[-static_cast<ptrdiff_t>(width)] : l;
                const int tl = h > 0 && w > 0 ? pixels[-static_cast<ptrdiff_t>(width) - 1] : t;
                const int predict = median(l, l + t - tl, t);

                pixels[0] = predict + (ziros == 0 ? next : 0);
                pixels++;

                if (ziros == 0)
                    std::tie(next, ziros) = decoder.decode();
                else
                    ziros--;
            }
        }
    }

    struct Header
    {
        uint32_t signature;
        uint32_t width;         // same for all channels
        uint32_t height;        // same for all channels
        uint8_t bits_per_pixel; // 8-16 supported
        uint32_t channel_nb;    // r,g,b,n1,n2,n3,n4...
    };

    std::vector<uint8_t> compressImage(const uint8_t *rgb, uint32_t width, uint32_t height, uint32_t channels)
    {
        auto planar = split_8bit_pixels_into_int16_channels(rgb, channels, width, height);
        auto out = std::vector(planar.size(), std::vector(0, uint8_t{}));
        std::vector<std::future<std::vector<uint8_t>>> futures;
        futures.reserve(channels);

        for (size_t i = 0; i < out.size(); ++i)
        {
            futures.push_back(
                std::async(true ? std::launch::deferred : std::launch::async, [&](size_t n)
                           { return encodeChannel(planar[n].data(), width, height); }, i));
        }

        Header hdr{signature, width, height, 8, channels};
        std::vector<uint8_t> hdr_and_channels;
        hdr_and_channels.reserve(sizeof(hdr) + width * height * channels * 3 / 4);
        auto bi = std::back_inserter(hdr_and_channels);
        std::copy(reinterpret_cast<const uint8_t *>(&hdr), reinterpret_cast<const uint8_t *>(&hdr) + sizeof(hdr), bi);

        for (size_t i = 0; i < futures.size(); ++i)
        {
            auto vec = futures[i].get();
            size_t vec_size = vec.size();
            std::copy(reinterpret_cast<const uint8_t *>(&vec_size), reinterpret_cast<const uint8_t *>(&vec_size) + sizeof(vec_size), bi);
            std::copy(vec.begin(), vec.end(), bi);
        }
        return hdr_and_channels;
    }

    RawImage8bit decompressImage(const uint8_t *begin, const uint8_t *end)
    {
        Header hdr{};
        std::copy(begin, begin + sizeof(hdr), reinterpret_cast<uint8_t *>(&hdr));
        begin += sizeof(hdr);
        std::vector<std::vector<int16_t>> planar(hdr.channel_nb, std::vector<int16_t>(hdr.width * hdr.height));
        std::vector<std::future<bool>> futures;
        futures.reserve(hdr.channel_nb);
        for (size_t i = 0; i < planar.size(); ++i)
        {
            size_t size;
            std::copy(begin, begin + sizeof(size), reinterpret_cast<uint8_t *>(&size));
            begin += sizeof(size);
            futures.push_back(
                std::async(true ? std::launch::deferred : std::launch::async, [&](const uint8_t *begin, const uint8_t *end, int16_t *pixels)
                           {
                               decodeChannel(hdr.width, hdr.height, begin, end, pixels);
                               return true;
                           },
                           begin, begin + size, planar[i].data()));
            begin += size;
        };
        for (auto &future : futures)
            future.get();

        return {join_int16_channels_to_8bit_pixels(planar, hdr.width, hdr.height, 0),
                hdr.width, hdr.height, hdr.channel_nb};
    }

}