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

        Writer(PagePool& pool) : pool(pool) {}


        inline int left_reserved() const {
            // if (res.size() == 0)
            //     return 0;
            return static_cast<int>(res.size())*64+static_cast<int>(left) - 64;
        }

        inline int reserved_total() const {
            return left_reserved();
        }

        void reserve(int count) {
            //count = std::max(count,32);
            while (left_reserved() < count) {
                res.put( pool.acquire_page() );
            }
        }

    inline void put_bits(uint32_t n, uint32_t value)
    {
        if (n == 0)
            return;
        assert(n <= left_reserved());
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
            assert(res.size() > 0);
            pool[ res.get() ] = (buf);
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

    struct Reader
    {
        uint64_t buf64{0}; // stores bits read from the buffer
        uint32_t buf32{0};
        uint32_t bits_valid_64{0}; // number of bits left in bits field
        int32_t bits_valid{0};
        const PagePool& pool;
        RingArray<size_t,4> res;
        Reader(const PagePool& pool) : pool(pool) {}

        inline int valid_reserved() const {

            return bits_valid + (res.size())*64;
        }

        inline  int reserved_total() const {
            return valid_reserved();
        }

        inline void reserve(uint32_t count) {
            bool glue_operation = bits_valid < 32;

            while (valid_reserved() < count)  {
               res.put( pool.get_next_read_page() );
            }
            if (glue_operation && res.size() > 0) {
                // if (res.size() == 0) {
                //     std::cout << "warning res.size()=0 has hole: bits_valid=" << bits_valid << " on_reserve=" << count << std::endl;
                //     return;
                // }
                //assert(res.size() > 0);
                assert(bits_valid_64 == 0);
                auto take = 32-bits_valid;
                //std::cout << " take = " << take << std::endl;
                assert(take > 0);
                bits_valid += 64;

                buf64 = (pool[ res.get() ]);
                bits_valid_64 = 64-take;
                buf32 |= static_cast<uint32_t>(buf64 >> (64-take));
                buf64 <<= take;

            }

        }

        inline void skip(uint32_t n)
        {
            assert(n <= 32);
            assert(bits_valid >= n);
            //std::cerr << "skip " << n << " " << peek_n(n) << std::endl;

            while (n > 0) {
                if (bits_valid_64 == 0) {
                    if ( res.size() == 0) {
                        assert(bits_valid >= n);

                        if (n == 32) {
                            buf32 = 0;
                            bits_valid -= 32;
                            assert(bits_valid == 0);
                            return;

                        }
                        assert(n < 32);
                        buf32 <<= n;
                        bits_valid -= n;
                        //std::cout << "bits_valid " << bits_valid << std::endl;
                        return;
                    } else {
                        buf64 = (pool[ res.get() ]);
                        bits_valid_64 = 64;
                        bits_valid += 64;
                    }
                }
                uint32_t take = std::min(n, bits_valid_64);
                buf32 = (take == 32) ? static_cast<uint32_t>(buf64 >> 32) : (buf32 << take | static_cast<uint32_t>(buf64 >> (64 - take)));
                buf64 <<= take;
                bits_valid_64 -= take;
                n -= take;
                bits_valid -= take;
                //pos += take;
            }
            //assert(bits_valid >= 32);
            // if (bits_valid < 32) {
            //     std::cout << "warn!! bits_valid < 32: " << bits_valid << " bits_valid_64=" << bits_valid_64 << std::endl;
            // }

        }

        inline uint32_t peek32()
        {
            assert(bits_valid  >= 32);
            return buf32;
        }
        inline uint32_t peek_n(uint32_t n)
        {
            assert(n != 0);
            assert(n <= 32);
            assert(bits_valid >= n);
            return n == 32 ? buf32 : (buf32 >> (32-n));
        }
    };

}


