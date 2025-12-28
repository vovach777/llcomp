#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <iterator>
#include <utility>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>


namespace Rice
{


    inline uint32_t to_unsigned(int32_t v)
    {
        // Используем статический каст к беззнаковому типу перед сдвигом,
        // чтобы избежать UB при сдвиге отрицательных чисел.
        return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31);
    }
    
    inline int32_t to_signed(uint32_t uv)
    {
        // Здесь логика верна: (uv >> 1) извлекает модуль, 
        // а (-(uv & 1)) создает маску из всех 1 (если нечетное) или всех 0 (если четное).
        return static_cast<int32_t>((uv >> 1) ^ -(uv & 1));
    }

namespace std20 {
template <class T>
int bit_width(T x) {
    using U = std::make_unsigned_t<T>;
    U v = static_cast<U>(x);
    if (v == 0) return 0;

    constexpr int bits = std::numeric_limits<U>::digits;

    if constexpr (sizeof(U) <= sizeof(unsigned int)) {
        return bits - __builtin_clz(v);
    } else if constexpr (sizeof(U) <= sizeof(unsigned long)) {
        return bits - __builtin_clzl(v);
    } else {
        return bits - __builtin_clzll(v);
    }
}
}




    inline void adapt_k(uint32_t value, uint32_t &k) {
    
        //k = std20::bit_width(value);
        if (value == 0) {
            k = 0;
            return;
        }
        //int value_of_k = k < 3? k : (((1 << k) ) >> 1) + 1;//k=0: 0; k=1: 1; k=2: 4/2+1=3; k=3: 8/2+1=5; k=4: 16/2+1=9; k=5: 32/2+1=17
        //k = std20::bit_width((value * 5 + value_of_k * 3) >> 3);
        //k = std20::bit_width((value * 2 + value / 2 + value_of_k + value_of_k/2+3) / 4);

        // uint32_t value_of_k = k < 2? k : ((1 << (k+1))) / 3;
        // k = std20::bit_width((value * 4 + value + value_of_k*3) / 8);
        uint32_t value_of_k = k < 4? k*2 : ((1 << (k+1))); //TODO: tune k=2,3
        k = std20::bit_width((value * 4 + value + value_of_k) / 8);
 
    }

    
    // inline void adapt_k(uint32_t value, uint32_t &k) {
        
    //     //int prev_k = k;
    //     const uint32_t k2 = std20::bit_width(value);
    //     if (k2 == 0){
    //         k = 0;
    //         return;
    //     }
    //     if ( k < k2) 
    //         k += std::clamp((k2-k)*3/5,1U,8U);
    //     else
    //     if ( k > k2 )
    //         k -= std::clamp( (k-k2)*3/5, 1U, 8U);

    //     //std::cout << value <<  " k: " << prev_k << " => " << k << std::endl;
    // }

    // Rice code: Encode unsigned integer x with parameter m
    template <typename Writer>
    void write(uint32_t x, uint32_t& m, Writer &&writer)
    {
        auto quotient = x >> m; 
        //std::cerr << quotient << " ";
        auto remainder = x & ((1 << m) - 1);
        if (quotient >= 32)
        {
            //std::cout << "!!! " << quotient << std::endl;
            writer.put_bits(32,0xffffffffu);
            writer.put_bits(32,x);
        } else {
            writer.put_bits(quotient + 1, ((1ULL << (quotient))-1) << 1); // Write quotient bits
            writer.put_bits(m, remainder);
            //std::cerr << remainder << " ";
        }
        adapt_k(x,m);
    }
    template <typename Reader>
    uint32_t read(uint32_t &m, Reader&& reader)
    {
        
        uint32_t res;
        if (reader.peek32() == 0xffffffffu)
        {
            reader.skip(32);
            res = reader.peek32();
            reader.skip(32);
        } else {
            uint32_t quotient = 0;
            auto ones_in_buffer = __builtin_clz(~reader.peek32());
            reader.skip(ones_in_buffer + 1);
            quotient += ones_in_buffer;
            if (m==0) {
                res=quotient;
            } else { 
                auto remainder = reader.peek32() >> (32-m);
                reader.skip(m);
                res = remainder + (quotient << m);
            }
        }
        adapt_k(res,m);
        return res;
    }
}

namespace RLGR {

constexpr bool RLMode = true;

class Encoder {
    
    uint32_t k=0;
    uint32_t rl=0;
    uint32_t kr=0;
    uint32_t kz=8;
    BitStream::Writer s;
    
    public:
    Encoder(BitStream::PagePool& pool) : s(pool) {}

    void put(int v) {
        if constexpr (RLMode) {
            if (k == 0) {   
                if (rl==0)   
                    s.reserve(128);          
                if (v == 0) {
                    rl+=1;
                    return;
                }
                Rice::write(rl,kr,s);
                Rice::write(v,kz,s);
                //std::cout << s.pos << std::endl;
                k = kz;
                rl = 0;
                return;
            }
        }
        s.reserve(64);
        Rice::write(v,k,s);
    }

    void flush() {
        if (rl > 0) {
            Rice::write(rl-1,kr,s);
            Rice::write(0,kz,s);
            k = kz;
            rl = 0;   
        }
        s.flush();
    }
};


class Decoder {
    uint32_t k=0;
    uint32_t kr=0;
    uint32_t kz=8;
    uint32_t rl{0};
    uint32_t val=0;
    BitStream::Reader s;
    public:
    Decoder(const BitStream::PagePool& pool) : s(pool) {}
    int get() {

        if constexpr (RLMode) {
            if (rl > 0) {
                rl--;
                return (rl == 0) ? val : 0;                                    
            }
            if (k == 0) {
                s.reserve(128);

                rl = Rice::read(kr,s);
                val =  Rice::read(kz,s);
                k = kz;
                if (rl == 0) {
                    return val;
                }
                return 0;
            }
        }
        s.reserve(64);
        auto r = Rice::read(k,s);
        return r;
    }

};
}


