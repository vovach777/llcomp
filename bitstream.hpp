#pragma once
//#define BUGFREE_CODE
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

struct Writer {
    alignas(8) uint64_t buf{0};
    uint32_t left{64};

/**
 * Write up to 32 bits into a bitstream.
 */

template <typename Flush>
inline void put_bits(uint32_t n, uint32_t value, Flush && flush)
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
        flush(reinterpret_cast<const uint8_t *>(&bswapped), 8);
        left += 64 - n;
        buf = value; //it's ok that have extra MSB rem bits.. it's cuts on flush later
    }
}

inline void byte_align() {
    uint32_t shift = left & 7;
    buf <<= shift;
    left -= shift;
}

template <typename Flush>
void flush(Flush && flush_f) {
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

struct Reader
{
    uint32_t buf32{0};
    uint64_t buf64{0}; // stores bits read from the buffer
    uint32_t bits_valid_64{0}; // number of bits left in bits field

    template <typename Refill>
    inline void skip(uint32_t n, Refill&& refill)
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
    template <typename Refill>
    inline void init(Refill&& refill)
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
    template <typename WriteBits>
    void write(uint32_t x, uint32_t m, WriteBits &&writeBits)
    {
#if 0
        auto quotient = x >> m; // Math.floor(x / (1 << m))
        auto remainder = x & ((1 << m) - 1);
        while (quotient--)
            writeBits(1,1);
        writeBits(1,0);
        writeBits(m, remainder);
#else
        auto quotient = x >> m; // Math.floor(x / (1 << m))
        auto remainder = x & ((1 << m) - 1);
        while (quotient >= 32)
        {
            writeBits(32,0xffffffffu);
            quotient -= 32;
        }
        writeBits(quotient + 1, ((1ULL << (quotient))-1) << 1); // Write quotient bits
        writeBits(m, remainder);
#endif
    }
    template <typename Peek32, typename Skip>
    uint32_t read(uint32_t m, Peek32 &&peek32, Skip &&skip)
    {
#if 0
        uint32_t quotient = 0;
        while (peek32() >> 31 == 1) {
            skip(1);
            quotient++;
        }
        skip(1);
        auto remainder = m == 0  ? 0 : ( peek32() >> (32 - m) );
        skip(m);
        return (quotient << m) | remainder;

#else
        uint32_t quotient = 0;
        while (peek32() == 0xffffffffu)
        {
            quotient += 32;
            skip(32);
        }
        auto ones_in_buffer = __builtin_clz(~peek32());
        skip(ones_in_buffer + 1);
        quotient += ones_in_buffer;
        if (m) {
            auto remainder = peek32() >> (32 - m);
            skip(m);
            return remainder + (quotient << m);
        }
        return quotient;

#endif
    }
}
