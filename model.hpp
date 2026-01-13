#pragma once
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
#include <iterator>
#include "polyfill.hpp"

template <size_t N = 16, bool reset_on_overflow = true, int threshold = 4, int bigthresh = 16, int max_nodes_lg2 = 16>
class DMCModel
{
public:
    static_assert(N == 16, "N=16 is only supported value now");
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

    void model_park() {
        state = 0;
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

using ModelK = DMCModel<16,false,2,2,12>;





