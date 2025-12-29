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
#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <smmintrin.h>
#include <unordered_set>
#include <numeric>
#include <optional>

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
    
    if (value == 0) {
        k = 0;
        return;
    }
    if (value > 0x10000 || k > 16) {
        auto k2 =  std20::bit_width(value);
        k = (5 * k2 + 3 * k+7) >> 3;        
        return;
    }

    int value_of_k = ((1 << k)-1)/2;
    k = std20::bit_width( (value*5 + value_of_k*3 + 0) >> 3);

    }


template <size_t N = 16, bool reset_on_overflow = true, int threshold = 4, int bigthresh = 16, int max_nodes_lg2 = 16>
class DMCModel
{
public:
    static_assert(max_nodes_lg2 <= 30, "max_nodes_lg2 must be <= 30");
    using IndexType = std::conditional_t<(max_nodes_lg2 <= 16), uint16_t, uint32_t>;
    struct NodeNext
    {
        IndexType next_offs[N];
    };
struct NodeCounter {
        uint64_t count{0};

        // Сумма всех счетчиков (SWAR)
        auto total() const {
            constexpr uint64_t NIB_MASK = 0x0F0F0F0F0F0F0F0FULL;
            auto x = count;
            // Складываем соседние полубайты: 4+4 бита -> в 8 бит
            x = (x & NIB_MASK) + ((x >> 4) & NIB_MASK);

            // Суммируем байты через умножение
            constexpr uint64_t MUL = 0x0101010101010101ULL;
            uint64_t t = x * MUL;
            return static_cast<uint8_t>(t >> 56);           
        }
        int unique() const {
            return __builtin_popcountll(count & 0x1111111111111111ULL * 0xFULL);
        }

        int operator [] (int symbol) const {
            return (count >> (symbol * 4)) & 0xF;
        }
        void  set_at(int symbol, int value) {
            count &= ~(0xFULL << (symbol*4));
            count |= (value+0ULL) << (symbol*4);
        }
        void mask_at(int symbol) {
            count &= ~(0xFULL << (symbol*4));
        }

        void update(int symbol) {            
            // Получаем текущее значение (0..15)
            int val = (count >> (symbol * 4)) & 0xF;
            if (val == 15) {
                // Масштабирование: делим все на 2 (сброс старшего бита каждого нимбла)
                // 15 (1111) -> 7 (0111). После инкремента станет 8.
                count = (count >> 1) & 0x7777777777777777ULL;
            }            
            // Безусловный инкремент. 
            count +=  1ULL << (symbol * 4);
        }

        template <typename AC>
        void encode_ac( int symbol, AC && ac ) const {
            auto count_copy = *this;
            int ctx=0;
            while (count_copy.count) {
                auto symbol_predict = count_copy.max_element();
                bool esc = symbol != symbol_predict;
                ac.put(esc,ctx++);
                if (!esc)
                    return;
                count_copy.mask_at(symbol_predict);
            }
            ac.put_bypass(symbol);
        }
        template <typename AC>
        int decode_ac(AC && ac) const {
            auto count_copy = *this;
            int ctx=0;
            while (count_copy.count) {                
                bool esc = ac.get(ctx++);
                auto symbol_predict = count_copy.max_element();
                if (!esc) {
                    return symbol_predict;
                }
                count_copy.mask_at(symbol_predict);
            }
            return ac.get_bypass();
        }

        int max_element() const {
            // 1. Загружаем count (исправил имя переменной)
            if ( count == 0) {             
                return 0;
            }
            __m128i v = _mm_cvtsi64_si128(count);

            __m128i mask = _mm_set1_epi8(0x0F);
            __m128i even = _mm_and_si128(v, mask);
            __m128i odd = _mm_and_si128(_mm_srli_epi64(v, 4), mask);
            __m128i bytes = _mm_unpacklo_epi8(even, odd);

            // Редукция максимума
            __m128i max_val = bytes;
            max_val = _mm_max_epu8(max_val, _mm_srli_si128(max_val, 8)); 
            max_val = _mm_max_epu8(max_val, _mm_srli_si128(max_val, 4)); 
            max_val = _mm_max_epu8(max_val, _mm_srli_si128(max_val, 2)); 
            max_val = _mm_max_epu8(max_val, _mm_srli_si128(max_val, 1)); 
            
            uint8_t m = _mm_extract_epi8(max_val, 0);
            __m128i cmp = _mm_cmpeq_epi8(bytes, _mm_set1_epi8(m));
            int mask_res = _mm_movemask_epi8(cmp);

            // mask_res не может быть 0, так как максимум всегда есть
            return __builtin_ctz(mask_res);
        }
    };

    bool is_full() const {
        return nodes_count >= get_nodes_max();
    }
    bool is_half() const {
        return nodes_count >= get_nodes_max()/2;
    }
    bool is_empty() const {
        return is_empty_flag;
    }

    DMCModel()
    {
        reset_model();
    }

    size_t get_nodes_count() const
    {
        return std::min(nodes_count, get_nodes_max());
    }

    constexpr size_t get_nodes_max() const
    {
        return 1ULL << max_nodes_lg2;
    }

    void reset_model()
    {
        is_empty_flag = true;
        nodes_counter.clear();
        nodes_counter.resize( 1ULL << max_nodes_lg2 );
        nodes_next.clear();
        nodes_next.resize( 1ULL << max_nodes_lg2 );        
        nodes_count = 0;
        auto state0 = nodes_count++;
        auto state1 = nodes_count++;        
        nodes_counter[state0].count = 0;
        std::fill(nodes_next[state0].next_offs, nodes_next[state0].next_offs + N, state1 );
        nodes_counter[state1].count = 0;
        std::fill(nodes_next[state1].next_offs, nodes_next[state1].next_offs + N, state0 );
        state = state0;
        cached_k = {};

        // state = nodes_count++;
        // nodes_counter[state].count = 1;
        // std::fill(nodes_next[state].next_offs, nodes_next[state].next_offs + N, state );

    }



    uint32_t get_k() {
        if (cached_k)  return *cached_k;
        cached_k = nodes_counter[state].max_element();
        return *cached_k;
    }
    uint32_t get_next_k() {

        return nodes_next[state].next_offs[ get_k() ].max_element();
    }
    uint32_t get_total() const {
        return nodes_counter[state].total();
    }
    uint32_t get_unique() const {
        return nodes_counter[state].unique();
    }


    void update(int b)
    {
        is_empty_flag = false;
        assert(static_cast<size_t>(b) < N);
        cached_k = std::nullopt;

        nodes_counter[state].update(b);
        auto next_b = nodes_next[state].next_offs[b];// state->next_offs[b];
        int next_b_total = nodes_counter[next_b].total();
        auto c0  = nodes_counter[state][b];
        if ( (c0 >= (threshold) &&
            next_b_total >= bigthresh + c0 ) )
        {

            if (nodes_count == get_nodes_max()) {
                if (reset_on_overflow)
                    reset_model();
            } else
            {
                auto new_ = nodes_count++;
                auto & new_counter = nodes_counter[new_];
                std::copy( nodes_next[next_b].next_offs, nodes_next[next_b].next_offs+N, nodes_next[new_].next_offs);
                int scale = (c0 << 16) / next_b_total; // fixed point 16.16
                assert(scale > 0);
                int max_detect{0};
                int max_index{0};
                assert( nodes_counter[next_b].count > 0);
                for (int i = 0; i < N; ++i)
                {
                    auto src_counter = nodes_counter[next_b][i];
                    int scaled = (src_counter * scale + 0x8000) >> 16;
                    auto remainder = src_counter - scaled;
                    new_counter.set_at(i,scaled);
                    assert( src_counter >= scaled );
                    nodes_counter[next_b].set_at(i,remainder );
                }                            
                nodes_next[state].next_offs[b] = new_;
            }
        }

        state = nodes_next[state].next_offs[b]; // state->next_offs[b] );

    }
    bool trust() const {
        // Доверяем модели, если средняя повторяемость символа >= 2
        auto t = get_total();
        auto u = get_unique();
        return t >= (u << 1);
    }
private:
    std::vector<NodeCounter> nodes_counter;
    std::vector<NodeNext> nodes_next;
    uint32_t state;
    size_t nodes_count{0};
    std::optional<int> cached_k{};
    
    bool is_empty_flag{true};
};

using ModelK = DMCModel<16,false,2,8,12>;

constexpr size_t  Interleave = 1;

template <typename T, uint32_t max_size_>
struct RingArray {
    static constexpr uint32_t max_size = max_size_;
    std::array<T,max_size_> buf_;

    constexpr void put(T item)
    {
        buf_[head_] = item;
        head_ = (head_ + 1) % max_size_;
        size_++;
    }

    constexpr T get() {
        //Read data and advance the tail (we now have a free space)
        auto val = buf_[tail_];
        tail_ = (tail_ + 1) % max_size_;
        size_--;

        return val;
                
    }
    constexpr uint32_t size() const {
        return size_;
    }
    constexpr bool empty() const {
        return size_ == 0;
    }
    constexpr bool full() const {
        return size_ == max_size_;
    }

    uint32_t head_ = 0;
    uint32_t tail_ = 0;
    uint32_t size_ = 0;
    
};

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
        std::cerr << quotient << " ";
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
    uint64_t buf{0};
    uint32_t left{64};
    uint64_t **pool;
    RingArray<uint64_t*,8> res;
    uint32_t pos{0};
    Writer(uint64_t ** pool) : pool(pool){}


int left_reserved() const { 
    // if (res.size() == 0) {
    //     return 0;
    // }
    // return left == 0 ? res.size()*64 : left + (res.size()-1)*64; 
    return static_cast<int>(res.size())*64+static_cast<int>(left) - 64;
}

    void reserve(int count) {
        while (left_reserved() < count)
           res.put( (*pool)++ ); 
        //std::cout << left_reserved() << std::endl;
    }

inline void put_bits(uint32_t n, uint32_t value)
{
    if (n == 0)
        return;  
    pos += n;
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
        *res.get() = bswap64(buf);
        left += 64 - n;
        buf = value; //it's ok that have extra MSB rem bits.. it's cuts on flush later        
    }
    //std::cout <<  left_reserved() << std::endl;
}

inline void byte_align() {
    uint32_t shift = left & 7;
    buf <<= shift;
    left -= shift;
}

void flush() {
    if (left < 64) {
        buf <<= left;
        assert(res.size() > 0);
        *res.get() = bswap64(buf);
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
    int pos{0};
    int credit{-1};//-1 : init,0 - no credit, >0  peek32 credit n from next reserve()
    const uint64_t **pool{nullptr};
    RingArray<const uint64_t*,8> res;
    Reader(const uint64_t **pool) : pool(pool) {}

    int valid_reserved() {
 
        return bits_valid + (res.size())*64;
    }

    void reserve(uint32_t count) {
        if (credit == -1) {
            buf64 = bswap64( *(*pool)++ );
            bits_valid_64 = 32;
            buf32 = buf64 >> 32;
            buf64 <<= 32;
            bits_valid = 64;
            credit = 0;
        }
        if (credit > 0) {
            assert(count >= 64);
            //assert(count >= credit);
            count -= 64; //один блок возьмем прямо сейчас
            assert(res.size() == 0);
            buf64 = bswap64(*(*pool)++);                
            bits_valid_64 = 64;
            bits_valid += 64;            
            int n = credit;
            credit = 0;
            skip(n); //lazy skip
            // std::cout << "afer credit return: " << bits_valid << " count=" << count << " valid_resrved()=" << valid_reserved() <<  std::endl; 
            // assert( credit == 0);
            // while (valid_reserved() < count) 
            //     res.put((*pool)++);
            // std::cout << "afer reserve: " << valid_reserved() << std::endl; 
            // return;


        }
        while (valid_reserved() < count) {
           res.put((*pool)++);
       }
       //std::cout << valid_reserved() << std::endl;
               

    }

    inline void skip(uint32_t n)
    {
        assert(n <= 32);
        assert(bits_valid >= 0);
        
        while (n > 0) {
            if (bits_valid_64 == 0) {
                if ( res.size() == 0) {
                    //std::cout << "+credit: " << bits_valid << " - " << n << std::endl;
                    credit = n;
                    return;
                }
                buf64 = bswap64(*res.get());                
                bits_valid_64 = 64;
                bits_valid += 64;
            }
            uint32_t take = std::min(n, bits_valid_64);
            buf32 = (take == 32) ? static_cast<uint32_t>(buf64 >> 32) : (buf32 << take | static_cast<uint32_t>(buf64 >> (64 - take)));
            buf64 <<= take;
            bits_valid_64 -= take;
            n -= take;
            bits_valid -= take;
            pos += take;

            //std::cout <<  valid_reserved() << std::endl;
        }
        assert(bits_valid >= 32);
        // if (bits_valid < 32) {
        //     std::cout << "warn!! bits_valid < 32: " << bits_valid << std::endl;
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
namespace SimpleLRGR {
constexpr uint32_t MAX_K = 240;
class Encoder {
    
    
    uint32_t rl=0;
    uint32_t k=0;
    uint32_t kr=0;
    ModelK model;
    BitStream::Writer s;

    
    public:
    Encoder(uint64_t**pool) : s(pool) {}

    void put(int v) {
        uint32_t k_ = k >> 3;
        uint32_t kr_ = kr >> 3;
        uint32_t kr__ = kr_;                
        if (k_ > 0) { 
            if (rl == 0)
                s.reserve(1+k_);   
            if (v == 0) {
               rl++;
               if (rl == (1u << k_)) {
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
                s.reserve(64);
                rl=0;
                k -= k > 6 ? 6 : k;
                //k_ = k >> 3;
                /* v is positive here */
                if (kr__ <= 15 &&  model.trust()) {
                    kr__ = model.get_k();
                }
                Rice::write(v-1,kr__,s);
                model.update( std::min(15, std20::bit_width(v)));
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
            if (kr__ <= 15 &&  model.trust()) {
                kr__ = model.get_k();
            }
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
            model.update( std::min(15, std20::bit_width(v)));
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
    size_t stream_size{0};
    ModelK model;
    public:
    Decoder(const uint64_t**pool, size_t stream_size=size_t(-1)) : s(pool), stream_size(stream_size) {}
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
            s.reserve(1+k_);
            run_mode = true;
            if ( s.peek_n(1) == 0) {
                s.skip(1);
                //полное окно                
                rl = 1 << k_;
                k += 4;
                if (k > MAX_K)
                k = MAX_K;
                k_ = k >> 3;
                assert( stream_size >= rl);
                stream_size -= rl;
                //full windows = sync point
                //s.reserve(1+k_);
                rl_val = 0;
                rl--;
                return 0; 
            } else {
                s.skip(1);
                rl = s.peek_n(k_);
                s.skip(k_);
                s.reserve(64);
                k -= k > 6 ? 6 : k;
                //k_ = k >> 3;
                /* v is positive here */
                assert( stream_size >= rl);
                stream_size -= rl;
                if (stream_size > 0) {
                    stream_size--;
                    if (kr__ <= 15 &&  model.trust()) {
                        kr__ = model.get_k();
                    }
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
                    model.update( std::min(15, std20::bit_width( rl_val ) ));
                } else {
                    std::cerr << "skip terminator (out of size)" << std::endl;
                    rl_val = 0; //virtual terminator
                }
                if (rl==0) {
                    //остатка нет - это значит что нулей нет и нужно сразу выдать rl_val
                    assert(stream_size > 0 && "read after eof");
                    run_mode = false;
                    return rl_val; //sync point
                } 
                rl--;
                return 0;//if rl == 0 : sync point
            }
        }        
        s.reserve(64);//max GR code length ESC+raw code = 64bit
        if (kr__ <= 15 &&  model.trust()) {
            kr__ = model.get_k();
        }

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
        model.update( std::min(15, std20::bit_width( v ) ));

        return v;
    }

};
}



auto encoder(const std::vector<uint32_t> & data) {
    
    std::vector<uint64_t> pages(1024);
    uint64_t * pool = pages.data();

    std::vector<SimpleLRGR::Encoder> codecs;

    for (int i = 0; i < Interleave; ++i) {
        codecs.emplace_back(&pool);
    }

    
    for (size_t i=0; i < data.size(); i+=1) {
        
        codecs[i % Interleave].put(data[i]);
       
    }
    
        
    std::cout << "---flush---" << std::endl;
    for (auto & codec : codecs)
       codec.flush();

    pages.resize(pool-pages.data());
    std::cout << "pack " <<  data.size() << " -> " << (pages.size()*8) << std::endl;

    std::cout << std::endl;
    
    return pages;

}
auto decoder(const std::vector<uint64_t>& pages, size_t count ) {
    auto b_begin = pages.data();    
    auto b_it = b_begin;
    auto b_end   = pages.data() +  pages.size();


    std::cout << "\n--- decoder ---\n";

    const uint64_t* pool=pages.data();
    std::vector<SimpleLRGR::Decoder> decoders; //(Interleave,SimpleLRGR::Decoder<RLMode>(&pool)); 

    for (int i = 0; i < Interleave; ++i) {
        decoders.emplace_back(&pool, (count / Interleave) + (i < (count % Interleave)));
    }

    std::vector<uint32_t> res(count);
    for (size_t i = 0; i < count; i+=1) {        
        res[i] = decoders[i % Interleave].get();
        //std::cout << int(res[i]) << std::endl;
        
    }
    return res;


}

constexpr size_t iterations = 256;

int main() {

    std::vector<uint32_t> stream;
    for (int i=0; i<iterations; ++i) 
    {
        //stream.push_back( (rand() % 32) == 0 ? rand()  : 0 );        
        stream.push_back( std::pow( std::abs( std::sin(double(i)/40.)),8.) * 0xffu);
    }

    size_t print_i = 0;
    for (auto v :stream) 
    {
        std::cout << " " << int(v);
        if (++print_i == 256){
            std::cout << "...";
            break;
        }
    }
    std::cout << std::endl;
    
    auto verify = decoder( encoder(stream), stream.size() );    


    // for (auto v :verify) 
    // {
    //     std::cout << " " << int(v);
    // }
    std::cout << std::endl;

    if (verify.size() < stream.size()) {
        std::cerr << "(verify.size() < stream.size())\n";
        return 1;
    }
    for (size_t i = 0; i < stream.size(); ++i ) 
    {
        if ( stream[i] != verify[i]) {
            std::cerr << "(stream[i] != verify[i])\n";
            return 2;
        }
        //std::cout << std::setw(4) << stream[i]
    }
    std::cout << "success!\n";

}
