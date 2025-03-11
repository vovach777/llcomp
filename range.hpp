#include <functional>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>

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
inline const std::vector<int> NEXT_STATE_MPS = {
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 123, 124, 125, 124, 125, 126, 127
};

inline const std::vector<int> NEXT_STATE_LPS = {
    1, 0, 0, 1, 2, 3, 4, 5, 4, 5, 8, 9, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 18, 19, 22,
    23, 22, 23, 24, 25, 26, 27, 26, 27, 30, 31, 30, 31, 32, 33, 32, 33, 36, 37, 36, 37, 38, 39, 38,
    39, 42, 43, 42, 43, 44, 45, 44, 45, 46, 47, 48, 49, 48, 49, 50, 51, 52, 53, 52, 53, 54, 55, 54,
    55, 56, 57, 58, 59, 58, 59, 60, 61, 60, 61, 60, 61, 62, 63, 64, 65, 64, 65, 66, 67, 66, 67, 66,
    67, 68, 69, 68, 69, 70, 71, 70, 71, 70, 71, 72, 73, 72, 73, 72, 73, 74, 75, 74, 75, 74, 75, 76,
    77, 76, 77, 126, 127
};

inline const std::vector<double> MPS_PROBABILITY = {
    0.5, 0.5273, 0.5504, 0.5735, 0.5945, 0.6155, 0.6366, 0.6555, 0.6723, 0.6891,
    0.7059, 0.7206, 0.7353, 0.7479, 0.7605, 0.7731, 0.7857, 0.7962, 0.8067,
    0.8172, 0.8256, 0.8361, 0.8445, 0.8529, 0.8592, 0.8676, 0.8739, 0.8803,
    0.8866, 0.8929, 0.8992, 0.9034, 0.9097, 0.9139, 0.9181, 0.9223, 0.9265,
    0.9307, 0.9349, 0.9391, 0.9412, 0.9454, 0.9475, 0.9517, 0.9538, 0.9559, 0.958,
    0.9601, 0.9622, 0.9643, 0.9664, 0.9685, 0.9706, 0.9727, 0.9748, 0.9748,
    0.9769, 0.979, 0.979, 0.9811, 0.9811, 0.9832, 0.9853, 1,
};

class Model3 {
public:
    Model3(size_t contexts_count = (11 * 11 * 11 * 5 * 5 + 1) / 2, size_t states_count = 8)
        : contexts_count(contexts_count), states_count(states_count),
          states(contexts_count * states_count, 0) {}
    void reset() {
        std::fill(states.begin(), states.end(), 0);
    }
    double P(size_t context, size_t bitpos) {
        size_t index = std::min(contexts_count - 1, context) * states_count + std::min(states_count - 1, bitpos);
        size_t state = states[index];
        bool mps = state & 1;
        double p_mps = MPS_PROBABILITY[state >> 1];
        return mps ? p_mps : 1 - p_mps;
    }

    void update(size_t context, size_t bitpos, bool bit) {
        assert(bit == 0 || bit == 1);
        size_t index = std::min(contexts_count - 1, context) * states_count + std::min(states_count - 1, bitpos);
        size_t state = states[index];
        states[index] = (state & 1) == bit ? NEXT_STATE_MPS[state] : NEXT_STATE_LPS[state];
    }
    bool is_large_context() {
        return contexts_count > 666;
    }

private:
    size_t contexts_count;
    size_t states_count;
    std::vector<size_t> states;
};

#define HAS_BUILTIN_CLZ
inline int ilog2_32(uint32_t v, int error_value = 0) {
    if (v == 0)
        return error_value;
#ifdef HAS_BUILTIN_CLZ
    return 31 - __builtin_clz(v);
#else
    return static_cast<uint32_t>(std::log2(v));
#endif
}


inline void putSymbol(int v, bool isSigned, std::function<void(int, bool)> putRac) {
    if (!putRac) {
        return;
    }

    if (v) {
        int a = std::abs(v);
        int e = ilog2_32(a,0);
        putRac(0, false);

        int ctx = 1;
        for (int i = 0; i < e; i++) {
            putRac(std::min(ctx++, 4), true);
        }
        putRac(std::min(ctx++, 4), false);

        ctx = 5;
        for (int i = e - 1; i >= 0; i--) {
            putRac(std::min(ctx++, 6), (a >> i) & 1);
        }

        if (isSigned) {
            putRac(7, v < 0);
        }
    } else {
        putRac(0, true);
    }
}

inline int getSymbol(bool isSigned, std::function<bool(int)> getRac) {
    if (!getRac) return 0;

    if (getRac(0)) return 0;

    int e = 0;
    int ctx = 1;
    while (getRac(std::min(ctx++, 4))) {
        e++;
        if (e > 31) {
            throw std::runtime_error("Invalid exponent");
        }
    }

    int value = 1;
    ctx = 5;
    for (int i = e - 1; i >= 0; i--) {
        value += value + getRac(std::min(ctx++, 6));
    }

    if (isSigned) {
        if (getRac(7))
            value = -value;
    }
    return value;
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

    writeU8(0x77);
    writeU8(channels);
    writeU16(width);
    writeU16(height);

    RangeEncoder comp([&](char x) {
        writeU8(x);
    });

    std::vector<std::vector<int16_t>> lines(3, std::vector<int16_t>(stride));
    Model3 model;
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
                if (model.is_large_context()) {
                    hash += quant5(L - l) * (5 * 11 * 11) + quant5(T - t) * (5 * 5 * 11 * 11);
                }
                const int predict = median(l, l + t - tl, t);
                int diff = (line0[x + i] - predict);

                if (hash < 0) {
                    hash = -hash;
                    diff = -diff;
                }
                assert(hash >= 0);

                putSymbol(diff, true, [&](int bitpos, bool bit) {
                    const double prob = model.P(hash, bitpos);
                    comp.put(bit, prob);
                    model.update(hash, bitpos, bit);
                });
            }
        }
    }
    comp.finish();
    buffer.resize(write_pos);
    //return std::vector<uint8_t>(buffer.begin(), buffer.begin() + write_pos);
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
    if ((magic & 0xFF) != 0x77) {
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

    Model3 model;
    const int x1 = channels;
    const int x2 = channels * 2;
    std::vector<std::vector<int16_t>> lines(3, std::vector<int16_t>(stride));

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

                if (model.is_large_context()) {
                    hash += quant5(L - l) * (5 * 11 * 11) + quant5(T - t) * (5 * 5 * 11 * 11);
                }

                const int predict = median(l, l + t - tl, t);

                bool neg_diff = false;
                if (hash < 0) {
                    hash = -hash;
                    neg_diff = true;
                }

                int diff = getSymbol(true, [&](int bitpos) {
                    const double prob = model.P(hash, bitpos);
                    bool bit = decomp.get(prob);
                    model.update(hash, bitpos, bit);
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
