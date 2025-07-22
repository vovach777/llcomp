#pragma once
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <iterator>
#include <cstdint>
#include <utility>
#include <cstring>

namespace BitStream {

#ifdef _MSC_VER
    inline constexpr uint32_t bswap32(uint32_t x)
    {
        return ((((x) << 8 & 0xff00) | ((x) >> 8 & 0x00ff)) << 16 | ((((x) >> 16) << 8 & 0xff00) | (((x) >> 16) >> 8 & 0x00ff)));
    }

    inline  constexpr uint64_t bswap64(uint64_t x)
    {
        return (uint64_t)bswap32(x) << 32 | bswap32(x >> 32);
    }
#else
    inline uint64_t bswap64(uint64_t x) {
        return __builtin_bswap64(x);
    }
#endif

template <typename Flush>
struct Writer {
    Flush flush_f;
    alignas(8) uint64_t buf{0};
    uint32_t left{64};
    Writer(Flush flush_f) : flush_f(flush_f) {}

/**
 * Write up to 32 bits into a bitstream.
 */

inline void put_bits(uint32_t n, uint32_t value)
{
    if (n == 0)
        return;
    assert(n <= 32);
    assert( (n == 32) || (value >> n) == 0 );
    if (n <= left) {
        buf <<= n;
        buf |= value;
        left -= n;
    } else {
        assert(left < 32);
        auto rem = n - left;
        buf <<= left;
        buf |= static_cast<uint64_t>(value) >> rem;
        alignas(8) uint64_t bswapped = bswap64(buf);
        flush_f(reinterpret_cast<const uint8_t *>(&bswapped), 8);
        left += 64 - n;
        buf = value; //it's ok that have extra MSB rem bits.. it's cuts on flush later
    }
}

inline void byte_align() {
    uint32_t shift = left & 7;
    buf <<= shift;
    left -= shift;
}

void flush() {
    if (left < 64) {
        buf <<= left;
        uint64_t bswapped = bswap64(buf);
        size_t nbytes = (64 - left + 7) >> 3;
        flush_f(reinterpret_cast<const uint8_t*>(&bswapped), nbytes);
        buf = 0;
        left = 64;
    }
}

};
template <typename Refill>
struct Reader
{
    Refill refill;
    uint32_t buf32{0};
    uint64_t buf64{0}; // stores bits read from the buffer
    uint32_t bits_valid_64{0}; // number of bits left in bits field
    Reader(Refill refill) : refill(refill) {}

    inline void skip(uint32_t n)
    {
        assert(n <= 32);

        while (n > 0) {
            if (bits_valid_64 == 0) {
                auto [data, len] = refill();
                if (len < 8) {
                    buf64 = 0;
                    std::memcpy(&buf64, data, len);
                    buf64 = bswap64(buf64);
                } else {
                    buf64 = bswap64(*reinterpret_cast<const uint64_t *>(data));
                }
                bits_valid_64 = 64;//no eof cheching! fill with ziros infinite
            }
            uint32_t take = std::min(n, bits_valid_64);
            buf32 = (take == 32) ? static_cast<uint32_t>(buf64 >> 32) : (buf32 << take | static_cast<uint32_t>(buf64 >> (64 - take)));
            buf64 <<= take;
            bits_valid_64 -= take;
            n -= take;
        }
    }
    inline void init()
    {
        auto [data, len] = refill();
        if (len < 8) {
            uint64_t tmp = 0;
            std::memcpy(&tmp, data, len);
            buf64 = bswap64(tmp);
        } else {
            buf64 = bswap64(*reinterpret_cast<const uint64_t *>(data));
        }
        buf32 = static_cast<uint32_t>(buf64 >> 32);
        buf64 <<= 32;
        bits_valid_64 = 32;
    }

    inline uint32_t peek32()
    {
        return buf32;
    }
    inline uint32_t peek_n(uint32_t n)
    {
        assert(n != 0);
        assert(n <= 32);

        return n == 32 ? buf32 : (buf32 >> (32-n));
    }
};

}


namespace Rice
{

    inline uint32_t to_unsigned(int32_t v)
    {
        return (v << 1) ^ (v >> 31);
    }

    inline int32_t to_signed(uint32_t uv)
    {
        return (uv >> 1) ^ -static_cast<int32_t>(uv & 1);
    }

    // Rice code: Encode unsigned integer x with parameter m
    template <typename Writer>
    void write(uint32_t x, uint32_t m, Writer &&writer)
    {
        auto quotient = x >> m; // Math.floor(x / (1 << m))
        auto remainder = x & ((1 << m) - 1);
        while (quotient >= 32)
        {
            writer.put_bits(32,0xffffffffu);
            quotient -= 32;
        }
        writer.put_bits(quotient + 1, ((1ULL << (quotient))-1) << 1); // Write quotient bits
        writer.put_bits(m, remainder);
    }
    template <typename Reader>
    uint32_t read(uint32_t m, Reader&& reader)
    {
        uint32_t quotient = 0;
        while (reader.peek32() == 0xffffffffu)
        {
            quotient += 32;
            reader.skip(32);
        }
        auto ones_in_buffer = __builtin_clz(~reader.peek32());
        reader.skip(ones_in_buffer + 1);
        quotient += ones_in_buffer;
        if (m) {
            auto remainder = reader.peek32() >> (32 - m);
            reader.skip(m);
            return remainder + (quotient << m);
        }
        return quotient;
    }
}
