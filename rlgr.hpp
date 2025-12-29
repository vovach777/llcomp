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


// Rice code: Encode unsigned integer x with parameter m
template <typename Writer>
void write(uint32_t x, uint32_t m, Writer &&writer)
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
    }
}

template <typename Reader>
uint32_t read(uint32_t m, Reader&& reader)
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
            auto remainder = reader.peek_n(m);
            reader.skip(m);
            res = remainder + (quotient << m);
        }
    }
    return res;
}
}


namespace RLGR {
constexpr uint32_t MAX_K = 240;
class Encoder {
    
    
    uint32_t rl=0;
    uint32_t k=0;
    uint32_t kr=0;
    BitStream::Writer s;

    
    public:
    Encoder(BitStream::PagePool& pool) : s(pool) {}

    void put(int v) {
        uint32_t k_ = k >> 3;
        uint32_t kr_ = kr >> 3;
        uint32_t kr__ = kr_;                
        if (k_ > 0) { 
            if (rl == 0)
                s.reserve(128);   
            if (v == 0) {
                
               rl++;
               if (rl == (1u << k_)) {
                   //model.update(0);
                  s.put_bits(1,0);// полное окно
                  rl=0;
                  k += 4;
                  if (k > MAX_K)
                    k = MAX_K;
                  k_ = k >> 3;
                  //s.reserve(1+k_); 
               }
            } else {                
                s.put_bits(1,1);
                s.put_bits(k_,rl);
                //s.reserve(64);
                rl=0;
                k -= k > 6 ? 6 : k;
                //k_ = k >> 3;
                /* v is positive here */
                // if (kr__ <= 15 &&  model.trust()) {
                //     kr__ = model.get_k();
                // }
                //std::cerr << " T";
                Rice::write(v-1,kr__,s);
                //model.update( std::min(15, std20::bit_width(v-1)));
                uint32_t vk = v >> kr_;
                if (vk == 0) {
                    kr -= kr > 2 ? 2 : kr;
                } else
                if (vk != 1) {
                    kr += vk;
                    if (kr > MAX_K) 
                        kr = MAX_K;
                }

            }
        } else {
            s.reserve(64);//max GR code length ESC+raw code = 64bit
            // if (kr__ <= 15 &&  model.trust()) {
            //     kr__ = model.get_k();
            // }
            Rice::write(v,kr__,s);            
            uint32_t vk = v >> kr_;
            if (vk == 0) {
                kr -= kr > 2 ? 2 : kr;
            } else
            if (vk != 1) {
                kr += vk;
                if (kr > MAX_K)
                    kr = MAX_K;
            }
            if (v == 0) {
                k += 3;
                if (k > MAX_K)
                    k = MAX_K;
            } else {
                k -= k > 3 ? 3 : k;
            }
            //model.update( std::min(15, std20::bit_width(v)));
        }
    }

    void flush() {
        if (rl > 0) {
            uint32_t k_ = k>>3;
            assert(k_ > 0);
            s.put_bits(1,1);
            s.put_bits(k_,rl);
        }
        s.flush();
        //std::cout << "model nodes = " <<  model.get_nodes_count() << std::endl;
    }
};

class Decoder {    
    uint32_t rl=static_cast<uint32_t>(-1);
    uint32_t rl_val{0};
    bool run_mode{false};
    uint32_t k=0;
    uint32_t kr=0;
    BitStream::Reader s;
    public:
    size_t stream_size{0};
    public:
    Decoder(const BitStream::PagePool& pool, size_t stream_size=size_t(-1)) : s(pool), stream_size(stream_size) {}
    
    int get() {        
        if (run_mode) {
            if (rl == 0 ) {
                run_mode = false;
                //rl_val == 0 -> полное окно. нет символа терминатора пока. 
                if (rl_val != 0)
                   return rl_val;
            } else {
                rl--;
                return 0;
            }
        }
        assert(stream_size > 0 && "read after eof");
        uint32_t k_ = k >> 3;
        uint32_t kr_ = kr >> 3;
        uint32_t kr__ = kr_;
        if (k_ > 0) {
            s.reserve(128);
            run_mode = true;
            if ( s.peek_n(1) == 0) {
                //model.update(0);
                s.skip(1);
                //полное окно                
                rl = 1 << k_;
                k += 4;
                if (k > MAX_K)
                    k = MAX_K;
                //k_ = k >> 3;
                assert( stream_size >= rl);
                stream_size -= rl;
                //full windows = sync point
                //s.reserve(1+k_);
                rl_val = 0;
                rl--;
                assert(rl > 0);
                return 0; 
            } else {
                s.skip(1);
                rl = s.peek_n(k_);
                s.skip(k_);
                //s.reserve(64);
                k -= k > 6 ? 6 : k;
                //k_ = k >> 3;
                /* v is positive here */
                assert( stream_size >= rl);
                stream_size -= rl;
                if (stream_size > 0) {
                    stream_size--;
                    // if (kr__ <= 15 &&  model.trust()) {
                    //     kr__ = model.get_k();
                    // }
                    rl_val = Rice::read(kr__,s) + 1;
                    uint32_t vk = rl_val >> kr_;
                    if (vk == 0) {
                        kr -= kr > 2 ? 2 : kr;
                    } else
                    if (vk != 1) {
                        kr += vk;
                        if (kr > MAX_K) 
                            kr = MAX_K;
                    }
                    //model.update( std::min(15, std20::bit_width( rl_val -1 ) ));
                } else {
                    //std::cerr << "skip terminator (out of size)" << std::endl;
                    rl_val = 0; //virtual terminator
                }
                if (rl==0) {
                    //остатка нет - это значит что нулей нет и нужно сразу выдать rl_val
                    assert(rl_val != 0 && "read after eof"); //rl=0 + out rl_val. if rl_val == 0 there is no value = but it must exist (host wona symbol)
                    run_mode = false;
                    return rl_val; //sync point
                } 
                rl--;
                return 0;//if rl == 0 : sync point
            }
        }        
        s.reserve(64);//max GR code length ESC+raw code = 64bit
        // if (kr__ <= 15 &&  model.trust()) {
        //     kr__ = model.get_k();
        // }

        uint32_t v = Rice::read(kr__,s);
        uint32_t vk = v >> kr_;
        if (vk == 0) {
            kr -= kr > 2 ? 2 : kr;
        } else
        if (vk != 1) {
            kr += vk;
            if (kr > MAX_K)
                kr = MAX_K;
        }
        if (v == 0) {
            k += 3;
            if (k > MAX_K)
                k = MAX_K;
        } else {
            k -= k > 3 ? 3 : k;
        }
        stream_size--;
        //model.update( std::min(15, std20::bit_width( v ) ));

        return v;
    }

};
}


