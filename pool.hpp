#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <stdexcept>

struct alignas(32) Chunk32 {
private:
    std::array<uint64_t,4> data_{};

public:
    using value_type = uint64_t;
    static constexpr size_t size() noexcept { return 4; }

    constexpr uint64_t& operator[](size_t i) noexcept {
        return data_[i];
    }

    constexpr const uint64_t& operator[](size_t i) const noexcept {
        return data_[i];
    }

    constexpr uint64_t* data() noexcept {
        return data_.data();
    }

    constexpr const uint64_t* data() const noexcept {
        return data_.data();
    }
};

class PagePool {
    private:
        std::vector<Chunk32> pages;
        size_t write_pos = 0;  // Текущая позиция записи (для BitWriter)
        mutable size_t read_pos = 0;   // Текущая позиция чтения (для BitReader)

    public:
        PagePool() = default;
        PagePool(std::vector<Chunk32> pages) : pages(pages), write_pos(pages.size()) {}

        size_t acquire_page() {
            pages.emplace_back();
            return write_pos++;
        }

        inline Chunk32& operator[](size_t index) {
            assert (index < pages.size() && "PagePool: index out of range");
            return pages[index];
        }

        inline const Chunk32& operator[](size_t index) const {
            assert (index < pages.size() && "PagePool: index out of range");
            return pages[index];
        }

        Chunk32 get_next_chunk() const {
            assert (read_pos < write_pos && "PagePool: no pages to read (sync error)");
            return pages[read_pos++];
        }

        // Получает следующую страницу для чтения
        size_t get_next_read_page() const {
            assert (read_pos < write_pos && "PagePool: no pages to read (sync error)");
            return read_pos++;
        }
        size_t size() const {
            return pages.size();
        }
        const Chunk32* data() const {
            return pages.data();
        }
        void reserve(size_t n) {
            pages.reserve(n);
        }

        auto move() const {
            return std::move( pages );
        }
    };
