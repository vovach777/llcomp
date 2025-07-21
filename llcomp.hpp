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
#include "bitstream.hpp"
#include "rlgr.hpp"

namespace llcomp {
constexpr inline auto ext = ".llrice";
constexpr inline uint8_t revision = 3;
constexpr inline uint8_t magic_revision = 0x33 + revision;
constexpr inline bool LargeModel = true;
constexpr inline size_t substates_nb = 1;
constexpr inline int default_k = 1;
constexpr inline int max_k = 15;
constexpr size_t getStatesNb() {
    if constexpr (LargeModel) {
        return (11 * 11 * 11 * 5 * 5 + 1) / 2 * substates_nb;
    } else {
        return (11 * 11 * 11 + 1) / 2 * substates_nb;
    }
}

inline const std::vector<int> quant5_table = {
    0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1,
};

inline const std::vector<int> quant11_table = {
    0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
    -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -4, -4,
    -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
    -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -1,
};

inline int quant11(int x) {
    return quant11_table[std::max(-128, std::min(127, x)) & 0xFF];
}

inline int quant5(int x) {
    return quant5_table[std::max(-128, std::min(127, x)) & 0xFF];
}

inline int median(int a, int b, int c) {
    if (a > b) {
        if (c > b) {
            if (c > a) b = a;
            else b = c;
        }
    } else {
        if (b > c) {
            if (c > a) b = c;
            else b = a;
        }
    }
    return b;
}

inline std::vector<uint8_t> compressImage(const std::vector<uint8_t>& rgb, int width, int height, int channels) {
    int size = width * height * channels;
    int stride = width * channels;
    assert(size == rgb.size());
    std::vector<uint8_t> buffer;
    buffer.reserve(size*3/4);
    size_t read_pos = 0;

    auto writeU8 = [&](uint8_t x) {
        buffer.push_back(x);
    };

    auto writeU16 = [&](uint16_t x) {
        buffer.push_back(x & 0xFF);
        buffer.push_back((x >> 8) & 0xFF);
    };

    writeU8(magic_revision);
    writeU8(channels);
    writeU16(width);
    writeU16(height);

    BitStream::Writer bit_stream{};
    auto flusher = [&](const uint8_t* data, size_t size) {
        buffer.resize(buffer.size() + size);
        if ( size == 8 )
            std::memcpy(buffer.data() + buffer.size() - 8, data, 8);
        else
            std::memcpy(buffer.data() + buffer.size() - size, data, size);
    };

    std::vector<std::vector<int16_t>> lines(3, std::vector<int16_t>(stride));
    std::vector<RLGR::Encoder> states( getStatesNb());
    int pos = 0;
    const int x1 = channels;
    const int x2 = channels * 2;

    for (int h = 0; h < height; ++h) {
        auto& line0 = lines[h % 3];
        auto& line1 = lines[(h + 3 - 1) % 3];
        auto& line2 = lines[(h + 3 - 2) % 3];
        for (int w = 0; w < width; ++w) {
            const int x = w * channels;
            if (channels >= 3) {
                int r = rgb[pos];
                int g = rgb[pos + 1];
                int b = rgb[pos + 2];
                b -= g;
                r -= g;
                g += (b + r) / 4;

                line0[x + 0] = r;
                line0[x + 1] = g;
                line0[x + 2] = b;
                for (int i = 3; i < channels; i++) {
                    line0[x + i] = rgb[pos + i];
                }
            } else {
                for (int i = 0; i < channels; i++) {
                    line0[x + i] = rgb[pos + i];
                }
            }
            pos += channels;
            for (int i = 0; i < channels; i++) {
                const int l = w > 0 ? line0[x - x1 + i] : h > 0 ? line1[x + i] : 128;
                const int t = h > 0 ? line1[x + i] : l;
                const int L = w > 1 ? line0[x - x2 + i] : l;
                const int tl = h > 0 && w > 0 ? line1[x - x1 + i] : t;
                const int tr = h > 0 && w < (width - 1) ? line1[x + x1 + i] : t;
                const int T = h > 1 ? line2[x + i] : t;

                int hash = (quant11(l - tl) +
                    quant11(tl - t) * (11) +
                    quant11(t - tr) * (11 * 11));
                if (LargeModel) {
                    hash += quant5(L - l) * (5 * 11 * 11) + quant5(T - t) * (5 * 5 * 11 * 11);
                }
                const int predict = median(l, l + t - tl, t);
                int diff = (line0[x + i] - predict);

                if (hash < 0) {
                    hash = -hash;
                    diff = -diff;
                }
                assert(hash >= 0);
                auto udiff = Rice::to_unsigned(diff);
                // uint8_t& k = states[hash];
                // Rice::write( udiff, k, [&](uint32_t n, uint32_t value){
                //     bit_stream.put_bits(n, value, flusher);
                // });
                // auto new_k = BitStream::bitsize(udiff);
                // k = std::min<uint8_t>(max_k, (new_k*2+k) / 3); // Адаптивное обновление k
                auto & state = states[hash];
                state.writeUnsigned(udiff,[&](uint32_t n, uint32_t value){
                         bit_stream.put_bits(n, value, flusher);
                     });
            }
        }
    }
    bit_stream.flush(flusher);
    return buffer;
}

struct RawImage {
    std::vector<uint8_t> pixels;
    uint16_t width;
    uint16_t height;
    uint8_t channels;
};

inline RawImage decompressImage(const std::vector<uint8_t>& data) {
    size_t pos = 0;
    const uint8_t magic = data[pos++];
    //assert((magic & 0xFF) == 0x77 && "Invalid magic number");
    if ((magic & 0xFF) != magic_revision) {
        throw std::runtime_error("Invalid magic number");
    }
    const uint8_t channels = data[pos++];
    const uint16_t width = (data[pos] | (data[pos + 1] << 8)); pos += 2;
    const uint16_t height = (data[pos] | (data[pos + 1] << 8)); pos += 2;
    const size_t stride = width * channels;
    std::vector<uint8_t> pixels(width * height * channels);
    size_t rgb_pos = 0;


    const int x1 = channels;
    const int x2 = channels * 2;
    std::vector<std::vector<int16_t>> lines(3, std::vector<int16_t>(stride));
    std::vector<RLGR::Decoder> states( getStatesNb() );
    BitStream::Reader bit_stream{};
    auto refiller = [&]() {
        if (pos + 8 <= data.size()) {
            pos += 8;
            return std::make_pair(data.data() + pos - 8, size_t{8});
        } else {
            size_t rem = data.size() - pos;
            pos += rem;
            return std::make_pair(data.data() + data.size() - rem ,  rem);
        }
    };
    bit_stream.init(refiller);

    for (size_t h = 0; h < height; ++h) {
        auto& line0 = lines[h % 3];
        auto& line1 = lines[(h + 2) % 3];
        auto& line2 = lines[(h + 1) % 3];

        for (size_t w = 0; w < width; ++w) {
            const size_t x = w * channels;
            for (size_t i = 0; i < channels; ++i) {
                const int l = w > 0 ? line0[x - x1 + i] : (h > 0 ? line1[x + i] : 128);
                const int t = h > 0 ? line1[x + i] : l;
                const int L = w > 1 ? line0[x - x2 + i] : l;
                const int tl = h > 0 && w > 0 ? line1[x - x1 + i] : t;
                const int tr = h > 0 && w < width - 1 ? line1[x + x1 + i] : t;
                const int T = h > 1 ? line2[x + i] : t;

                int hash = (quant11(l - tl) +
                    quant11(tl - t) * 11 +
                    quant11(t - tr) * 11 * 11);

                if (LargeModel) {
                    hash += quant5(L - l) * (5 * 11 * 11) + quant5(T - t) * (5 * 5 * 11 * 11);
                }

                const int predict = median(l, l + t - tl, t);

                bool neg_diff = false;
                if (hash < 0) {
                    hash = -hash;
                    neg_diff = true;
                }
                auto& state = states[hash];
                auto udiff = state.readUnsigned( [&](){ return bit_stream.peek32();}, [&](uint32_t n) { bit_stream.skip(n, refiller); });
                // auto new_k = BitStream::bitsize(udiff);
                // k = std::min<uint8_t>(max_k, (k * 2 + new_k + 2) / 3);
                auto diff = Rice::to_signed(udiff);

                if (neg_diff) {
                    diff = -diff;
                }
                line0[x + i] = predict + diff;
            }

            int r = line0[x + 0];
            int g = line0[x + 1];
            int b = line0[x + 2];
            g -= ((r + b) / 4);
            r += g;
            b += g;
            pixels[rgb_pos++] = std::max(0, std::min(255, r));
            pixels[rgb_pos++] = std::max(0, std::min(255, g));
            pixels[rgb_pos++] = std::max(0, std::min(255, b));
            for (size_t i = 3; i < channels; ++i) {
                pixels[rgb_pos++] = line0[x + i];
            }
        }
    }
    return {std::move(pixels), width, height, channels};
}
}
