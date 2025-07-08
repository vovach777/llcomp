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

namespace llcomp {
constexpr inline auto ext = ".llcomp";
constexpr inline uint8_t revision = 1;
constexpr inline uint8_t magic_revision = 0x77 + revision;
constexpr inline bool LargeModel = true;
constexpr inline int param_e_lim = 4;  //0,1,2,3,4
constexpr inline int param_r_lim = 6;  //5,6
constexpr inline int param_s_bit = 7;  //7
constexpr inline int substates_nb = 8; //=
constexpr size_t getStatesNb() {
    if constexpr (LargeModel) {
        return (11 * 11 * 11 * 5 * 5 + 1) / 2 * substates_nb;
    } else {
        return (11 * 11 * 11 + 1) / 2 * substates_nb;
    }
}
class RangeEncoder {
public:
    RangeEncoder(std::function<void(uint8_t)> put_byte): low(0), range(0xFF00), put_byte(put_byte), outstanding_count(0), outstanding_byte(-1) {
    }

    void renorm_encoder() {
        while (range < 0x100) {
            if (outstanding_byte < 0) {
                outstanding_byte = low >> 8;
            } else if (low <= 0xFF00) {
                put_byte(outstanding_byte);
                for (; outstanding_count; outstanding_count--)
                    put_byte(0xFF);
                outstanding_byte = low >> 8;
            } else if (low >= 0x10000) {
                put_byte(outstanding_byte + 1);
                for (; outstanding_count; outstanding_count--)
                    put_byte(0x00);
                outstanding_byte = (low >> 8) & 0xFF;
            } else {
                outstanding_count++;
            }
            low = (low & 0xFF) << 8;
            range <<= 8;
        }
    }

    void put(bool bit, double probability) {
        int range1 = std::max(1, std::min(range - 1, static_cast<int>(range * probability)));
        assert(range1 < range);
        assert(range1 > 0);
        if (!bit) {
            range -= range1;
        } else {
            low += range - range1;
            range = range1;
        }
        assert(range >= 1);
        renorm_encoder();
    }

    void finish() {
        range = 0xFF;
        low += 0xFF;
        renorm_encoder();
        range = 0xFF;
        renorm_encoder();
    }

private:
    int outstanding_count;
    int outstanding_byte;
    int low;
    int range;
    std::function<void(char)> put_byte;
};

class RangeDecoder {
public:
    RangeDecoder(std::function<uint8_t()> get_byte) : get_byte(get_byte), range(0xFF00) {
        low = this->get_byte() << 8;
        low |= this->get_byte();
    }

    void refill() {
        if (range < 0x100) {
            range <<= 8;
            low <<= 8;
            low += get_byte();
        }
    }

    bool get(double probability) {
        int range1 = std::max(1, std::min(range - 1, static_cast<int>(range * probability)));
        range -= range1;
        if (low < range) {
            refill();
            return 0;
        } else {
            low -= range;
            range = range1;
            refill();
            return 1;
        }
    }

private:
    int range;
    int low;
    std::function<int()> get_byte;
};

namespace binarization {
    #define HAS_BUILTIN_CLZ
    constexpr uint32_t UnsafeBehavior = 0xFFFFFFFFU; // Example value, adjust as needed
    template <uint32_t error_value = 0, typename T>
    inline uint32_t ilog2_32(T v) {
        if constexpr (!std::is_integral_v<T>) {
            throw std::invalid_argument("ilog2_32 requires an integral type");
        }
        if constexpr (error_value != UnsafeBehavior) {
            //required value protection
            if constexpr (std::is_signed_v<T>) {
                if (v <= 0)
                    return error_value;
            } else {
                if (v == 0)
                    return error_value;
            }
        }
    #ifdef HAS_BUILTIN_CLZ
        return 31 - __builtin_clz(static_cast<uint32_t>(v));
    #else
        return static_cast<uint32_t>(std::log2(v));
    #endif
    }

    /**
     * @brief Encodes a symbol using a binary arithmetic coding scheme.
     *
     * @param isSigned Indicates whether the value `v` is signed. If true, the sign of `v` is encoded.
     * @param e_limit The maximum value for the exponent context. Default is 4.
     * @param r_limit The maximum value for the remainder context. Default is 7.
     * @param sign_ctx The context for the sign bit. Default is 7.
     * @param v The value to encode.
     * @param putRac A callback function to handle the binary arithmetic coding.
     *               It takes two parameters: the context (int) and the binary value (bool).
     *               If the callback is null, the function does nothing.
     */
    template <bool isSigned, int e_limit=4, int r_limit=7, int sign_ctx=7, typename T>
    inline void putSymbol(T v, std::function<void(int, bool)> putRac) {

        if constexpr (!std::is_integral_v<T>) {
            throw std::invalid_argument("putSymbol requires an integral type");
        }

        if (!putRac) {
            return;
        }
        uint32_t uv;
        if constexpr (std::is_signed_v<T>) {
            uv = static_cast<uint32_t>(std::abs(v));
        } else {
            uv = static_cast<uint32_t>(v);
        }

        if (uv != 0) {
            auto e = ilog2_32<UnsafeBehavior>(uv);
            assert(e != 0);

            putRac(0, false);

            int ctx = 1;
            for (int i = 0; i < e; i++) {
                putRac(std::min(ctx++, e_limit), true);
            }
            putRac(std::min(ctx, e_limit), false);

            ctx = e_limit + 1;
            for (int i = e - 1; i >= 0; i--) {
                putRac(std::min(ctx++, r_limit), (uv >> i) & 1);
            }

            if constexpr (isSigned) {
                putRac(sign_ctx, v < 0);
            }
        } else {
            putRac(0, true);
        }
    }

    /**
     * @brief Decodes a symbol using a binary arithmetic coding scheme.
     * @param isSigned Indicates whether the decoded value should be treated as signed.
     * @param e_limit The maximum value for the exponent context. Default is 4.
     * @param r_limit The maximum value for the remainder context. Default is 7.
     * @param sign_ctx The context for the sign bit. Default is 7.
     * @param getRac A function that returns a boolean value based on the context.
     *               It takes an integer context as input and returns a boolean value.
     * @return The decoded symbol as an integer.
     * @throws std::runtime_error If the decoded exponent exceeds 31, indicating invalid data.
     */
    template <bool isSigned, int e_limit=4, int r_limit=7, int sign_ctx=7>
    inline auto getSymbol(std::function<bool(int)> getRac) {

        std::conditional_t<isSigned, int32_t, uint32_t> value{0};
        if (!getRac) return value; // If the callback is null, return zero.

        if (getRac(0)) return value;

        int e = 0;
        int ctx = 1;
        value = 1;
        while (getRac(std::min(ctx++, e_limit))) {
            e++;
            if (e > 31) {
                throw std::runtime_error("Invalid exponent");
            }
        }

        ctx = e_limit + 1;
        for (int i = e - 1; i >= 0; i--) {
            value += value + getRac(std::min(ctx++, r_limit));
        }

        if constexpr (isSigned) {
            if (getRac(sign_ctx))
                value = -value;
        }
        return value;
    }
}

namespace cabac {

    constexpr inline const auto nextStateMps = std::array{
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
        119, 120, 121, 122, 123, 124, 125, 124, 125, 126, 127
    };

    constexpr inline const auto nextStateLps = std::array{
        1, 0, 0, 1, 2, 3, 4, 5, 4, 5, 8, 9, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 18, 19, 22,
        23, 22, 23, 24, 25, 26, 27, 26, 27, 30, 31, 30, 31, 32, 33, 32, 33, 36, 37, 36, 37, 38, 39, 38,
        39, 42, 43, 42, 43, 44, 45, 44, 45, 46, 47, 48, 49, 48, 49, 50, 51, 52, 53, 52, 53, 54, 55, 54,
        55, 56, 57, 58, 59, 58, 59, 60, 61, 60, 61, 60, 61, 62, 63, 64, 65, 64, 65, 66, 67, 66, 67, 66,
        67, 68, 69, 68, 69, 70, 71, 70, 71, 70, 71, 72, 73, 72, 73, 72, 73, 74, 75, 74, 75, 74, 75, 76,
        77, 76, 77, 126, 127
    };

    constexpr inline const auto mpsProbability = std::array{
            0.5156,0.5405,0.5615,0.5825,0.6016,0.6207,0.6398,0.6570,
            0.6723,0.6875,0.7028,0.7162,0.7295,0.7410,0.7525,0.7639,
            0.7754,0.7849,0.7945,0.8040,0.8117,0.8212,0.8289,0.8365,
            0.8422,0.8499,0.8556,0.8613,0.8671,0.8728,0.8785,0.8823,
            0.8881,0.8919,0.8957,0.8995,0.9033,0.9072,0.9110,0.9148,
            0.9167,0.9205,0.9224,0.9263,0.9282,0.9301,0.9320,0.9339,
            0.9358,0.9377,0.9396,0.9415,0.9434,0.9454,0.9473,0.9473,
            0.9492,0.9511,0.9511,0.9530,0.9530,0.9549,0.9568,0.9702
        };

        struct State {
            uint8_t state{0}; //0.5 probability is default state
            constexpr bool mps_bit() const { return state & 1; }
            constexpr double P() const { return mps_bit() ? mpsProbability[state >> 1] : 1.0 - mpsProbability[state >> 1]; }
            constexpr operator double() const {
                return P();
            }
            constexpr void update(bool bit) {
                state = (bit == mps_bit()) ? nextStateMps[state] : nextStateLps[state];
            }
        };
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
    std::vector<uint8_t> buffer(size);
    int write_pos = 0;
    int read_pos = 0;

    auto writeU8 = [&](uint8_t x) {
        buffer[write_pos++] = x;
    };

    auto writeU16 = [&](uint16_t x) {
        buffer[write_pos++] = x & 0xFF;
        buffer[write_pos++] = (x >> 8) & 0xFF;
    };

    writeU8(magic_revision);
    writeU8(channels);
    writeU16(width);
    writeU16(height);

    RangeEncoder comp([&](char x) {
        writeU8(x);
    });

    std::vector<std::vector<int16_t>> lines(3, std::vector<int16_t>(stride));
    std::vector<cabac::State> states( getStatesNb() );
    int pos = 0;
    const int x1 = channels;
    const int x2 = channels * 2;
    int Y_period = 0;
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

                binarization::putSymbol<true,param_e_lim,param_r_lim,param_s_bit>(diff,[&](int ctx, bool bit) {
                   auto base = states.begin() +  hash * substates_nb;
                   auto& state = base[ctx];
                   comp.put(bit, state.P());
                   state.update(bit);
                });

            }
        }
    }
    comp.finish();
    buffer.resize(write_pos);
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

    RangeDecoder decomp([&]() -> uint8_t {
        if (pos >= data.size())
            return 0;
        return data[pos++];
    });

    const int x1 = channels;
    const int x2 = channels * 2;
    std::vector<std::vector<int16_t>> lines(3, std::vector<int16_t>(stride));
    std::vector<cabac::State> states( getStatesNb() );

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

                int diff = binarization::getSymbol<true,param_e_lim,param_r_lim, param_s_bit>([&](int ctx) {
                    auto base = states.begin() +  hash * substates_nb;
                    auto& state = base[ctx];
                    bool bit = decomp.get(state.P());
                    state.update(bit);
                    return bit;
                 });


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
