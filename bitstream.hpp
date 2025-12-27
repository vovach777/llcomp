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
        

class PagePool {
private:
    std::vector<uint64_t> pages;
    size_t write_pos = 0;  // Текущая позиция записи (для BitWriter)
    mutable size_t read_pos = 0;   // Текущая позиция чтения (для BitReader)

public:
    PagePool() = default;
    PagePool(std::vector<uint64_t> pages) : pages(pages), write_pos(pages.size()) {}

    size_t acquire_page() {
        pages.emplace_back(0);
        return write_pos++;
    }

    uint64_t& operator[](size_t index) {
        if (index >= pages.size()) {
            throw std::out_of_range("PagePool: index out of range");
        }
        return pages[index];
    }

    uint64_t operator[](size_t index) const {
        if (index >= pages.size()) {
            throw std::out_of_range("PagePool: index out of range");
        }
        return pages[index];
    }

    // Получает следующую страницу для чтения
    size_t get_next_read_page() const {
        if (read_pos >= write_pos) {
            throw std::runtime_error("PagePool: no pages to read (sync error)");
        }        
        return read_pos++;
    }
    size_t size() const {
        return pages.size();
    }
    const uint64_t* data() const {
        return pages.data();
    }
    void reserve(size_t n) {
        pages.reserve(n);
    }

    std::vector<uint64_t> move() const {
        return std::move( pages );
    }
};


template <typename T, size_t N = 4>
class RingArray {
private:
    std::array<T, N> buffer;
    size_t head = 0;
    size_t tail = 0;
    size_t count = 0;

public:
    // Добавляет элемент в буфер
    bool put(const T& item) {
        if (count == N) return false;
        buffer[tail] = item;
        tail = (tail + 1) % N;
        ++count;
        return true;
    }

    // Извлекает элемент из буфера
    bool get(T& item) {
        if (count == 0) return false;
        item = buffer[head];
        head = (head + 1) % N;
        --count;
        return true;
    }

    bool peek(T& item) const {
        if (count == 0) return false;
        item = buffer[head];
        return true;
    }

    size_t size() const { return count; }
    bool empty() const { return count == 0; }
    bool full() const { return count == N; }
};


class Writer {
private:
    PagePool& pool;
    RingArray<size_t, 4> res;  // Хранит индексы страниц
    uint64_t buf = 0;
    int left = 64;  // Количество свободных бит в buf

    // Вычисляет количество доступных бит
    size_t available_bits() const {        
        return res.size() * 64 + left - 64;
    }

public:
    Writer(PagePool& pool) : pool(pool) {}

    // Резервирует не менее N бит
    void reserve(size_t N) {
        while (available_bits() < N) {
            if (res.full()) {
                throw std::runtime_error("BitWriter::reserve ring buffer is full (sync error)");
            }
            size_t page_idx = pool.acquire_page();
            res.put(page_idx);
        }        
    }

    // Запись n бит из value (n <= 32)
    void put_bits(int n, uint32_t value) {
        assert(n <= 32 && "put_bits: n must be <= 32");
        assert( (n == 32) || (value >> n) == 0 );
        if (n <= left) {
            buf <<= n;
            buf |= value;
            left -= n; //если left == 0 : flush сбросит
        } else {
            auto rem = n - left;
            buf <<= left;
            buf |= static_cast<uint64_t>(value) >> rem;
            size_t page_idx; 
            if (!res.get(page_idx)) {
                throw std::runtime_error("BitWriter::put_bits no reserved pages (sync error)");
            }
            pool[page_idx] = bswap64(buf);
            left += 64 - n;
            buf = value; //тут мы не делаем  `& ((1 << rem) - 1)` потому что мусорные биты будут выталкнуты из 64 бит регистра
        }
    }

    // Сброс оставшихся бит в пул
    void flush() {
        if (left < 64) {
            // Страницы уже зарезервированы через reserve(N),
            // поэтому просто берём следующую страницу из кольца
            size_t page_idx;
            if (!res.get(page_idx)) {
                throw std::runtime_error("BitWriter::flush no reserved pages (sync error)");
            }
            buf <<= left;
            pool[page_idx] = bswap64(buf);
            buf = 0;
            left = 64;
        }
    }
};



class Reader {
private:
    const PagePool& pool;
    RingArray<size_t, 4> res;  // Хранит индексы страниц
    uint64_t buf = 0;
    int bits_valid = 0;  // Количество значимых бит в buf (0..64)

    size_t available_bits() const {
        return bits_valid + res.size() * 64;
    }

public:
    Reader(const PagePool& pool) : pool(pool) {}

    void reserve(size_t N) {
        while (available_bits() < N) {
            if (res.full()) {
                throw std::runtime_error("BitReader::reserve ring buffer is full (sync error)");
            }
            size_t page_idx = pool.get_next_read_page();
            res.put(page_idx);
        }
        if (bits_valid == 0) {
            size_t idx;
            if (!res.get(idx)) {
                throw std::runtime_error("BitReader::reserve  out of reserve!");
            }
            buf = bswap64(pool[idx]);
            bits_valid = 64;
        }
    }

    // peek32() — просмотр следующих 32 бит без изменения состояния.
    // НИКАКИХ reserve() внутри. Если не хватает зарезервированных бит — падение.
    uint32_t peek32() {
        assert(available_bits() >= 32 && "BitReader::peek32: insufficient reserve");
        if (bits_valid < 32) {
            size_t idx;
            if (!res.peek(idx)) {
                assert(false && "BitReader::peek32 out of reserve!");
            }
            uint64_t next  = bswap64(pool[idx]);
            uint32_t take  = 32-bits_valid;
            uint64_t chunk = (next >> (64-take));            
            buf |= chunk << 32;
        }
        return buf >> 32;
    }

    inline uint32_t peek_n(uint32_t n)
    {
        assert(n != 0);
        assert(n <= 32);
        return peek32() >> (32-n);
    }

    void skip(int n) {
        if (n <= bits_valid) {
            bits_valid -= n;
            buf <<= n;
        } else {
            n -= bits_valid;
            size_t idx;
            if (!res.get(idx)) {
                throw std::runtime_error("BitReader::skip out of reserve!");
            }
            buf = __builtin_bswap64(pool[idx]) << n;
            bits_valid = 64-n;
        }        
    }
};


}