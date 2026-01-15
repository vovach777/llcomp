#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

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
            assert (index < pages.size() && "PagePool: index out of range");
            return pages[index];
        }

        uint64_t operator[](size_t index) const {
            assert (index < pages.size() && "PagePool: index out of range");
            return pages[index];
        }

        uint64_t get_next_read_page() const {
            assert (read_pos < write_pos && "PagePool: no pages to read (sync error)");
            return pages[read_pos++];
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
