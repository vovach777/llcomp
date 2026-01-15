#pragma once
#include <cstdint>

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

    inline uint64_t bswap64(uint64_t x) noexcept {
        #if defined(__GNUC__) || defined(__clang__)
            return __builtin_bswap64(x);
        #elif defined(_MSC_VER)
            return _byteswap_uint64(x);
        #else
            return ((x & 0x00000000000000FFull) << 56) |
                   ((x & 0x000000000000FF00ull) << 40) |
                   ((x & 0x0000000000FF0000ull) << 24) |
                   ((x & 0x00000000FF000000ull) << 8) |
                   ((x & 0x000000FF00000000ull) >> 8) |
                   ((x & 0x0000FF0000000000ull) >> 24) |
                   ((x & 0x00FF000000000000ull) >> 40) |
                   ((x & 0xFF00000000000000ull) >> 56);
         #endif
    }

    inline uint16_t bswap16(uint16_t x) noexcept {
        #if defined(__GNUC__) || defined(__clang__)
            return __builtin_bswap16(x);
        #elif defined(_MSC_VER)
            return _byteswap_ushort(x);
        #else
            return (x << 8) | (x >> 8);
        #endif
    }

#if defined(__GNUC__) || defined(__clang__)
    #define likely(x)       __builtin_expect(!!(x), 1)
    #define unlikely(x)     __builtin_expect(!!(x), 0)
#else
    // Для MSVC или других компиляторов, которые это не поддерживают
    #define likely(x)       (x)
    #define unlikely(x)     (x)
#endif

// // Подсказки для предсказателя переходов (Branch Predictor)
// #if defined(__GNUC__) || defined(__clang__)
//     #define LIKELY(x) __builtin_expect(!!(x), 1)
//     #define UNLIKELY(x) __builtin_expect(!!(x), 0)
// #else
//     #define LIKELY(x) (x)
//     #define UNLIKELY(x) (x)
// #endif