#pragma once
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <iterator>
#include <cstdint>
#include <utility>
#include <cstring>
#include "polyfill.hpp"
#include "pool.hpp"
#include "ring.hpp"

namespace BitStream {

    struct Writer {
        uint64_t buf{0};
        uint32_t left{64};
        PagePool& pool;
        RingArray<size_t,4> res;
        int _left_reserved{0};

        Writer(PagePool& pool) : pool(pool) {}


        void reserve(int count) {
            //count = std::max(count,32);
            while (_left_reserved < count) {
                res.put( pool.acquire_page() );
                _left_reserved += 64;
            }
        }

    inline void put_bits(uint32_t n, uint32_t value)
    {
        if (n == 0)
            return;
        assert(n <= _left_reserved);
        assert(n <= 32);
        assert( (n == 32) || (value >> n) == 0 );
        if (n <= left) {
            buf <<= n;
            buf |= value;
            left -= n;
            _left_reserved -= n;
        } else {
            assert(left < 32);
            auto rem = n - left;
            buf <<= left;
            buf |= static_cast<uint64_t>(value) >> rem;
            assert(res.size() > 0);
            pool[ res.get() ] = (buf);
            _left_reserved -= n;
            left += 64 - n;
            buf = value; //it's ok that have extra MSB rem bits.. it's cuts on flush later
        }
    }

    inline void byte_align() {
        uint32_t shift = left & 7;
        buf <<= shift;
        left -= shift;
    }

    inline void flush() {
        if (left < 64) {
            buf <<= left;
            assert(res.size() > 0);
            pool[ res.get() ] = (buf);
            buf = 0;
            left = 64;
        }
    }

    };

    struct alignas(32) Reader
    {
        uint64_t buf64{0}; // stores bits read from the buffer
        uint32_t buf32{0};
        uint32_t bits_valid_64{0}; // number of bits left in bits field
        int32_t total_buffered{0};  // cnt32 + cnt64 + res.size()*64
        RingArray<uint64_t,4> res;
        const PagePool& pool;
        Reader(const PagePool& pool) : pool(pool) {}

        inline void reserve(uint32_t count) {
            auto glue_operation_if_below_32 = total_buffered;

            while (total_buffered < count)  {
               res.put( pool.get_next_read_page() );
               total_buffered += 64;
            }
            if (glue_operation_if_below_32 < 32 && res.size() > 0) {
                // if (res.size() == 0) {
                //     std::cout << "warning res.size()=0 has hole: bits_valid=" << bits_valid << " on_reserve=" << count << std::endl;
                //     return;
                // }
                //assert(res.size() > 0);
                assert(bits_valid_64 == 0);
                auto take = 32-glue_operation_if_below_32;
                //std::cout << " take = " << take << std::endl;
                assert(take > 0);

                buf64 = res.get();
                bits_valid_64 = 64-take;
                buf32 |= static_cast<uint32_t>(buf64 >> (64-take));
                buf64 <<= take;
            }

        }

        inline void skip(uint32_t n)
        {
            assert(n != 0);
            assert(n <= 32);
            assert(total_buffered >= n);
            if (likely( n <= bits_valid_64)) {
                total_buffered -= n;
                if (unlikely(n == 32)){
                    buf32 = static_cast<uint32_t>(buf64 >> 32);
                    buf64 <<= 32;
                    bits_valid_64 -= 32;
                }else {
                    buf32 <<= n;
                    buf32 |= static_cast<uint32_t>(buf64 >> (64 - n));
                    buf64 <<= n;
                    bits_valid_64 -= n;
                }
                return;
            }

            if (bits_valid_64 > 0) {
                const uint32_t rem{bits_valid_64};
                bits_valid_64 = 0;
                buf32 <<= rem;
                buf32 |= static_cast<uint32_t>(buf64 >> (64U - rem));
                n -= rem;
                total_buffered -= rem;
            }
            assert( n != 0 );
            total_buffered -= n;

            if ( unlikely(res.size() == 0)) {
                if (unlikely(n==32))
                    buf32 = 0;
                 else
                    buf32 <<= n;
                return;
            } else {
                buf64 = res.get();
                bits_valid_64 = 64;
                if (unlikely(n == 32)){
                    buf32 = static_cast<uint32_t>(buf64 >> 32);
                    buf64 <<= 32;
                    bits_valid_64 -= 32;
                }else {
                    buf32 <<= n;
                    buf32 |= static_cast<uint32_t>(buf64 >> (64 - n));
                    buf64 <<= n;
                    bits_valid_64 -= n;
                }
            }

        }

        inline uint32_t peek32()
        {
            assert(total_buffered  >= 32);
            return buf32;
        }
        inline uint32_t peek_n(uint32_t n)
        {
            assert(n != 0);
            assert(n <= 32);
            assert(total_buffered >= n);
            return n == 32 ? buf32 : (buf32 >> (32-n));
        }
    };

}


