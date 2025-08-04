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

template <typename Flush, typename Reserve, typename Tag>
struct Writer {
    uint64_t buf{0};
    uint32_t left{64};
    Flush flush_f;
    Reserve reserve_f;
    Tag tag;
    Writer(Flush flush_f, Reserve reserve_f, Tag tag) : flush_f(flush_f), reserve_f(reserve_f), tag(tag) {}

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
        flush_f(bswap64(buf), tag);
        left += 64 - n;
        buf = value; //it's ok that have extra MSB rem bits.. it's cuts on flush later
    }
    if (left < 32 && left + n >= 32) {
        reserve_f(tag);
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
        flush_f(bswap64(buf), tag);
        buf = 0;
        left = 64;
    }
}

};
template <typename Refill, typename Tag>
struct Reader
{
    uint64_t buf64{0}; // stores bits read from the buffer
    uint32_t buf32{0};
    uint32_t bits_valid_64{0}; // number of bits left in bits field
    Refill refill;
    Tag tag;
    Reader(Refill refill, Tag tag) : refill(refill), tag(tag) {}

    inline void skip(uint32_t n)
    {
        assert(n <= 32);

        while (n > 0) {
            if (bits_valid_64 == 0) {
                buf64 = bswap64(refill(tag));
                bits_valid_64 = 64;
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
            buf64 = bswap64( refill(tag) );
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

