#pragma once
#include <array>
#include <cstdint>
#include <cassert>

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

