#pragma once
#include <cstdint>
#define USE_ESC_BITS 32

namespace Rice
{
    inline uint32_t to_unsigned(int32_t v)
    {
        return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31);
    }

    inline int32_t to_signed(uint32_t uv)
    {
        return static_cast<int32_t>((uv >> 1) ^ -(uv & 1));
    }

    // Rice code: Encode unsigned integer x with parameter m
    template <typename Writer>
    inline void write(uint32_t x, uint32_t m, Writer &&writer)
    {
        auto quotient = x >> m;
        auto remainder = x & ((1 << m) - 1);
        if (quotient >= 32)
        {
            writer.put_bits(32,0xffffffffu);
            #if USE_ESC_BITS != 32
            writer.put_bits(USE_ESC_BITS,x);
            #else
            writer.put_bits(32,x);
            #endif
        } else {
            writer.put_bits(quotient + 1, ((1ULL << (quotient))-1) << 1); // Write quotient bits
            writer.put_bits(m, remainder);
        }
    }
    template <typename Reader>
    inline uint32_t read(uint32_t m, Reader&& reader)
    {
        uint32_t res;

        if (reader.peek32() == 0xffffffffu)
        {
            reader.skip(32);
#if USE_ESC_BITS != 32
            res = reader.peek_n(USE_ESC_BITS);
            reader.skip(USE_ESC_BITS);
#else
            res = reader.peek32();
            reader.skip(32);
#endif
        } else {
            uint32_t quotient = 0;
            auto ones_in_buffer = __builtin_clz(~reader.peek32());
            reader.skip(ones_in_buffer + 1);
            quotient += ones_in_buffer;
            if (m==0) {
                res=quotient;
            } else {
                auto remainder = reader.peek_n(m);
                reader.skip(m);
                res = remainder + (quotient << m);
            }
        }
        return res;
    }
}
