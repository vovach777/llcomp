#include <cmath>
#include <algorithm>
#include <cstdint>
#include <ctime>
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

constexpr size_t  Interleave = 7;

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
        //Доверяем модели, если средняя повторяемость символа >= 2
        auto t = get_total();
        auto u = get_unique();
        return (t >= (u << 1)) && (t >= threshold) && (nodes_count > 8);
    }
private:
    std::vector<NodeCounter> nodes_counter;
    std::vector<NodeNext> nodes_next;
    uint32_t state;
    size_t nodes_count{0};
    std::optional<int> cached_k{};
    
    bool is_empty_flag{true};
};

using ModelK = DMCModel<16,false,2,4,12>;


template <typename T, uint32_t max_size_>
struct RingArray {
    static constexpr uint32_t max_size = max_size_;
    std::array<T,max_size_> buf_;

    constexpr void put(T item)
    {
        assert(!full());
        buf_[head_] = item;
        head_ = (head_ + 1) % max_size_;
        size_++;
    }

    constexpr T get() {
         assert(!empty());
        //Read data and advance the tail (we now have a free space)
        auto val = buf_[tail_];
        tail_ = (tail_ + 1) % max_size_;
        size_--;

        return val;
                
    }

    constexpr T peek() const {
         assert(!empty());
        auto val = buf_[tail_];
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
        //std::cerr << "GR " << x  << " m=" << m << std::endl;
        auto quotient = x >> m; 
        //std::cerr << quotient << " ";
        auto remainder = x & ((1 << m) - 1);
        if (quotient >= 32)
        {
            //std::cerr << "ESC! x=" << std::hex << x << std::dec << std::endl;
            //std::cout << "!!! " << quotient << std::endl; 
            writer.put_bits(32,0xffffffffu);
            writer.put_bits(32,x);
            //std::cerr << " GR " << 32+32 << std::endl;
        } else {
               //std::cerr << " GR " << quotient + 1 + m << std::endl;
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
            //std::cerr << " GR 64" << std::endl;
            reader.skip(32);
            res = reader.peek32();
            reader.skip(32);
            //std::cerr << "ESC!" << " bits_valid=" << reader.bits_valid << " x=" << std::hex << res << std::dec << std::endl;
        } else {
            uint32_t quotient = 0;
            //std::cerr << std::hex << reader.peek32() << " " << std::dec << 0 << std::endl; 
            auto ones_in_buffer = __builtin_clz(~reader.peek32());
            //std::cerr << " GR " << ones_in_buffer + 1 + m << std::endl;
            reader.skip(ones_in_buffer + 1);
          
            quotient += ones_in_buffer;
            if (m==0) {
                res=quotient;
            } else { 
                auto remainder = reader.peek_n(m);
                reader.skip(m);
                res = remainder + (quotient << m);
            }
            //std::cerr << "GR " << res << " m=" << m << std::endl; 
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

inline uint32_t g_id{0};
struct Writer {
    uint64_t buf{0};
    uint32_t left{64};
    uint64_t **pool;
    RingArray<uint64_t*,32> res;
    uint32_t pos{0};
    uint32_t id{0};
    const uint64_t* get_accum_pool() const {
        return res.empty() ? nullptr : res.peek();
    }

    Writer(uint64_t ** pool) : pool(pool), id(g_id++){}


    int left_reserved() const { 
        return static_cast<int>(res.size())*64+static_cast<int>(left) - 64;
    }

    int reserved_total() const {
        return left_reserved();
    }

    void reserve(int count) {
        //count = std::max(count,32);
        while (left_reserved() < count)
           res.put( (*pool)++ ); 
    }

inline void put_bits(uint32_t n, uint32_t value)
{
    if (n == 0)
        return;  
    //std::cerr << " put " << n << " " << value << std::endl;
    if (n > left_reserved()) {
            //std::cerr << std::stacktrace::current() << std::endl;
        std::cerr << "Error: not enough reserved bits in BitStream::Writer. Requested " << n << ", but only " << left_reserved() << " available." << std::endl;
    }
    assert(n <= left_reserved());
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
    //int credit{-1};//-1 : init,0 - no credit, >0  peek32 credit n from next reserve()
    const uint64_t **pool{nullptr};
    RingArray<const uint64_t*,4> res;
    Reader(const uint64_t **pool) : pool(pool) {}

    int valid_reserved() const {
 
        return bits_valid + (res.size())*64;
    }

    int reserved_total() const {
        return valid_reserved();
    }



    const uint64_t* accum_pool{nullptr};

    const uint64_t* get_accum_pool() const {
        return accum_pool;
    }

    void reserve(uint32_t count) {
        bool glue_operation = bits_valid < 32;
     
        while (valid_reserved() < count)  {
            //assert(credit == 0);
           res.put((*pool)++);
        }
        if (glue_operation ) {
           
            //assert(credit == 0);
            if (res.size() == 0) {
                std::cout << "warning res.size()=0 has hole: valid bits=" << bits_valid << " on reserve=" << count << std::endl;
                return;
            }
            //assert(res.size() > 0);
            assert(bits_valid_64 == 0);
            auto take = 32-bits_valid;
            //std::cout << " take = " << take << std::endl;
            assert(take > 0);
            bits_valid += 64;  
            accum_pool = res.peek();
            buf64 = bswap64(*res.get());
            bits_valid_64 = 64-take;  

            
          
            //std::cout << std::hex << buf32 << std::dec << std::endl;
                
            buf32 |= static_cast<uint32_t>(buf64 >> (64-take));
            buf64 <<= take;               
    
            
            //std::cout << " valid_reserved=" << valid_reserved() << " bits_valid=" << bits_valid << " bits_valid_64=" << bits_valid_64 << std::endl;
        }

        //credit = 0;

    }

    inline void skip(uint32_t n)
    {
        assert(n <= 32);
        assert(bits_valid >= n);
        //std::cerr << "skip " << n << " " << peek_n(n) << std::endl;
        pos += n;
        
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
                    accum_pool = res.peek();
                    buf64 = bswap64(*res.get());                
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
        if (bits_valid < 32) {
            //std::cout << "warn!! bits_valid < 32: " << bits_valid << " bits_valid_64=" << bits_valid_64 << std::endl;
        }
        
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


namespace SimpleRLGR {
constexpr uint32_t MAX_K = 240;
class Encoder { 
    uint32_t rl=0;
    uint32_t k=0;
    uint32_t kr=0;
    ModelK model;
    bool inrun{false};
    BitStream::Writer s; 
    public:
    Encoder(uint64_t**pool) : s(pool) {
    }

    void reserve(int res) {
        s.reserve(res); //syncpoint before
    }

    void put(uint32_t v) {

        uint32_t k_ = k >> 3;
        uint32_t kr_ = kr >> 3;
        uint32_t kr__ = kr_;                
        if (k_ > 0) { 
                if (!inrun) {
                    reserve(1+k_);
                    inrun = true;
                }
            if (v == 0) {                               
               rl++; 
               if (rl == (1u << k_)) {
                 assert(rl != 1);
                  model.update(0);
                  s.put_bits(1,0);// полное окно
                  rl=0;
                  k += 4;
                  if (k > MAX_K)
                    k = MAX_K;
                  k_ = k >> 3;
                  //когда декодер вычитает этот бит полного окна - он обязать ждать rl-шагов прежде чем сделать этот резерв
                  reserve(1+k_);//k может вырастить тут - мы не будем ее считать - просто заразервируем 2 бита
               }
            } else {        
                inrun = false;  
                s.put_bits(1,1);                
                s.put_bits(k_,rl);
                //тут уже нет резерва.
                //тут можно сразу сказать декодеру сколько нужно брать из полубайты
                //после того как он выдаст все свои нули. если декодер сразу прочитает 
                //терминатор, то он не попадёт в своё окно. окно будет тут, а это будущее для декодера
                
                //std::cerr << "RL " << rl << std::endl;
           
                k -= k > 6 ? 6 : k;
                if (kr__ <= 15 &&  model.trust()) {
                    kr__ = model.get_k();
                } 
               reserve(64);
               rl=0; 
                v--;
                Rice::write(v,kr__,s);
                model.update( std::min(15, std20::bit_width(v)));
                uint32_t vk = (v) >> kr_;
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
            rl=0;
            reserve(64); 
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
            k_ = k >> 3;
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
    }
};

class Decoder {    
    uint32_t rl=0;
    bool run_mode{false};
    uint32_t k=0;
    uint32_t kr=0;   
    ModelK model;
    bool fullrl{false};
    BitStream::Reader s;

    public:
     size_t stream_size{0}; //control good ending

    Decoder(const uint64_t**pool, size_t stream_size=size_t(-1)) : s(pool), stream_size(stream_size) {}

    void reserve(int res) {
        s.reserve(res); //syncpoint before
    }

    uint32_t readRL(bool terminator=false) {
        uint32_t kr_ = kr >> 3;
        uint32_t kr__ = kr_;
        if (kr__ <= 15 &&  model.trust()) {
            kr__ = model.get_k();
        }
        auto term = Rice::read(kr__,s);
        uint32_t vk = term >> kr_;
        if (vk == 0) {
            kr -= kr > 2 ? 2 : kr;
        } else
        if (vk != 1) {
            kr += vk;
            if (kr > MAX_K) 
                kr = MAX_K;
        }
        model.update( std::min(15, std20::bit_width(term) ));
        return term+terminator;
    }

    uint32_t get() {       
        if (run_mode) {
            //rl_entry:
            if (fullrl) {
                assert(rl > 0);
                rl--;
                if (rl == 0) {
                    k += 4;
                    if (k > MAX_K)
                       k = MAX_K;
                    auto k_ = k >> 3;
                    reserve(1+k_);
                    run_mode = false;
                }
                stream_size--;
                return 0;
            } else {
                if (rl == 0) {
                    rl_0:
                    reserve(64);
                     k -= k > 6 ? 6 : k;
                     auto t = readRL(true);
                    run_mode = false;
                    stream_size--;
                    return t; 
                }
                assert(rl > 0);
                rl--;
                stream_size--;
                return 0;
            } 
        }
        assert(stream_size > 0 && "read after eof");
        uint32_t k_ = k >> 3;
        if (k_ > 0) {
            assert(rl == 0); 
            if (!fullrl) {         
                 reserve(1+k_);
            }
            fullrl = s.peek_n(1) == 0;
            s.skip(1);
            run_mode = true;
            if (fullrl) {                              
                model.update(0);
                //полное окно                
                rl = 1 << k_;
                assert(rl > 0);
            } else {
                rl = s.peek_n(k_);
                assert( stream_size >= rl);

                s.skip(k_);
                // if (rl == 0) {
                //     //std::cout << "*";
                //     goto rl_0;
                // }
         
            }
            //goto rl_entry;
            return get();
        }
        reserve(64);
        uint32_t v = readRL();
        if (v == 0) {
            k += 3;
            if (k > MAX_K)
                k = MAX_K;
        } else {
            k -= k > 3 ? 3 : k;
        }
        stream_size--;
        return v;
    }

};

}



auto encoder(const std::vector<uint32_t> & data) {
    
    std::vector<uint64_t> pages(0x8000);
    uint64_t * pool = pages.data();

    std::vector<SimpleRLGR::Encoder> codecs;

    for (int i = 0; i < Interleave; ++i) {
        auto & codec = codecs.emplace_back(&pool);
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

    std::cout << "\n--- decoder ---\n" << std::flush;
    std::cerr << "\n--- decoder ---\n" << std::flush;
 

    const uint64_t* pool=pages.data();
    std::vector<SimpleRLGR::Decoder> decoders; //(Interleave,SimpleLRGR::Decoder<RLMode>(&pool)); 

    for (int i = 0; i < Interleave; ++i) {         
       auto& decoder = decoders.emplace_back(&pool, (count / Interleave) + (i < (count % Interleave)));
    }

    std::vector<uint32_t> res(count);
    for (size_t i = 0; i < count; i+=1) {        
        res[i] = decoders[i % Interleave].get();
        //std::cout << int(res[i]) << std::endl;
        
    }
    return res;


}

constexpr size_t iterations = 4000;

int main() {
    //auto _srand=1767546614;
    auto _srand=time(0);
    std::cout << "_srand=" << _srand << std::endl;
    srand(_srand);
    std::vector<uint32_t> stream;
    
    // Config for each channel: {zero_probability_threshold (0-100), max_value_bits}
    struct ChannelConfig {
        int zero_prob;
        int max_bits;
    };
    
    std::array<ChannelConfig, 7> configs = {{
        {80, 8},   // Channel 0: 10% zeros, small values
        {10, 8},  // Channel 1: 50% zeros, medium values
        {80, 8},  // Channel 2: 90% zeros, large values
        {98,  16},   // Channel 3: 5% zeros, very small values
        {0, 16},  // Channel 4: 95% zeros, very large values
        {98, 16},  // Channel 5: 30% zeros
        {50, 12}   // Channel 6: 70% zeros
    }};

    #if 1
    for (int i=0; i<iterations; ++i) 
    {
        int channel = i % Interleave;
        auto& cfg = configs[channel];
        
        if ((rand() % 100) < cfg.zero_prob) {
             stream.push_back(0);
        } else {
             stream.push_back( rand() % (1 << cfg.max_bits) );
        }
    }
    #endif

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

    std::cerr << std::flush;


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
        //std::cout << std::setw(4) << stream[i];
    }
    std::cout << std::endl;
    std::cout << "success!\n";

}
