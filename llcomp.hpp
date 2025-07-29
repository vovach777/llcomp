#pragma once
#include <functional>
#include <cassert>
#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <string_view>
#include <cstdint>
#include <functional>
#include <utility>
#include <type_traits>
#include <future>
#include <limits>
#include <memory>
#include <type_traits>
#include <nmmintrin.h>
#include <sstream>
#include <fstream>

#include "bitstream.hpp"
#include "rlgr.hpp"

namespace llcomp
{
    constexpr inline auto ext = ".llr";
    constexpr inline uint8_t rev = '0';
    constexpr inline uint32_t signature = 'l' | 'l' << 8U | 'r' << 16U | rev << 24U;

    inline int median(int a, int b, int c)
    {
        if (a > b)
        {
            if (c > b)
            {
                if (c > a)
                    b = a;
                else
                    b = c;
            }
        }
        else
        {
            if (b > c)
            {
                if (c > a)
                    b = c;
                else
                    b = a;
            }
        }
        return b;
    }


    template <int src_depth, int dest_depth, typename T>
    auto split_rgb_into_planar(const T* data, int channels, int width, int height)
    {
        using DestType = std::conditional_t<(dest_depth > 15), int32_t, int16_t>;

        auto res = std::vector<std::vector<DestType>>(channels, std::vector<DestType>(width * height));
        size_t pos{0};
        constexpr int max_src_value = (1 << src_depth) - 1;

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                if (channels >= 3) {
                    int r = data[0];
                    int g = data[1];
                    int b = data[2];

                    if (r > max_src_value || g > max_src_value || b > max_src_value) {
                        throw std::invalid_argument("wrong source depth!");
                    }

                    if constexpr (dest_depth < src_depth) {
                        int shift = src_depth - dest_depth;
                        r >>= shift;
                        g >>= shift;
                        b >>= shift;
                    }

                    b -= g;
                    r -= g;
                    g += (b + r) / 4;

                    res[0][pos] = static_cast<DestType>(r);
                    res[1][pos] = static_cast<DestType>(g);
                    res[2][pos] = static_cast<DestType>(b);

                    for (int i = 3; i < channels; i++) {
                        res[i][pos] = static_cast<DestType>(data[i]);
                    }
                } else {
                    for (int i = 0; i < channels; i++) {
                        res[i][pos] = static_cast<DestType>(data[i]);
                    }
                }

                data += channels;
                pos += 1;
            }
        }

        return res;
    }


template <typename S, typename T>
void join_planar_into_rgb(const std::vector<std::vector<S>> & channels, uint32_t width, uint32_t height, T* dest) {

    size_t pos = 0;
    for (uint32_t h=0; h < height; ++h)
    {
        for (uint32_t w=0; w < width; ++w)
        {
            if (channels.size() >= 3) {
                int r = channels[0][pos];
                int g = channels[1][pos];
                int b = channels[2][pos];
                g -= ((r + b) / 4);
                r += g;
                b += g;
                dest[0] = std::clamp<int32_t>(r,  std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
                dest[1] = std::clamp<int32_t>(g,  std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
                dest[2] = std::clamp<int32_t>(b,  std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
            }
            for (size_t n=3; n < channels.size(); ++n) {
                dest[n] = std::clamp<int32_t>(channels[n][pos],  std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
            }
            dest += channels.size();
            pos++;
        }
    }
}

    template <typename T>
    inline std::vector<uint8_t> encodeChannel(const T*data, uint32_t width, uint32_t height)
    {
        auto size = size_t(width) * height;
        std::vector<uint8_t> buffer;
        buffer.reserve(size * 3 / 4);
        BitStream::Writer bit_stream{[&](const uint8_t *data, size_t size)
                                     {
                                         buffer.resize(buffer.size() + size);
                                         if (size == 8)
                                             std::memcpy(buffer.data() + buffer.size() - 8, data, 8);
                                         else
                                             std::memcpy(buffer.data() + buffer.size() - size, data, size);
                                     }};
        RLGR::Encoder encoder{[&](uint32_t n, uint32_t val)
                              { bit_stream.put_bits(n, val); }};
        for (uint32_t h = 0; h < height; ++h)
        {
            for (uint32_t w = 0; w < width; ++w)
            {
                const int l = w > 0 ? data[-1] : h > 0 ? data[-ptrdiff_t(width)]
                                                       : 0;
                const int t = h > 0 ? data[-ptrdiff_t(width)] : l;
                const int tl = h > 0 && w > 0 ? data[-ptrdiff_t(width) - 1] : t;
                const int predict = median(l, l + t - tl, t);
                int diff = (static_cast<int>(data[0]) - predict);
                encoder.encode(diff);
                data++;
            }
        }
        encoder.encode(-1);
        bit_stream.flush();
        return buffer;
    }

    template<typename F>
    class Defer {
        F func;
    public:
        Defer(F&& f) : func(std::forward<F>(f)) {}
        ~Defer() { func(); }
    };

    template<typename F>
    Defer<F> make_defer(F&& f) {
        return Defer<F>(std::forward<F>(f));
    }

    struct RawImage
    {
        std::unique_ptr<uint8_t[]> data{};
        uint32_t width{};
        uint32_t height{};
        uint32_t channel_nb{};
        uint8_t bits_per_channel{};
        enum class ChannelType{ none, uint8, uint16, uint16be, int32 };
        ChannelType channel_type{ ChannelType::none };

        template<typename T>
        T* as() {
            static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t>,
                "Unsupported type in RawImage::as()");
            return reinterpret_cast<T*>(data.get());
        }
        template<typename T>
        const T* as() const {
            static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t>,
                "Unsupported type in RawImage::as()");
            return reinterpret_cast<const T*>(data.get());
        }

        static inline auto get_next_line(std::ifstream& in,  bool check_comment = false)
        {
            for (;;)
            {
                std::string line{};
                if (!(in.eof() || in.fail()))
                {
                    std::getline(in, line);
                    if (!line.empty() && check_comment && line[0] == '#')
                        continue;
                }
                return line;
            }
        }

        static inline uint32_t lzcnt_s(uint32_t x)
        {
            return (x) ? __builtin_clz(x) : 32;
        }

        static void swap_bytes_16bit_sse(uint16_t* data, size_t count) {
            size_t i = 0;
            size_t simd_count = count & ~7; // по 8 uint16_t = 16 байт = 128 бит

            const __m128i shuffle_mask = _mm_set_epi8(
                14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1
            );

            for (; i < simd_count; i += 8) {
                __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
                v = _mm_shuffle_epi8(v, shuffle_mask);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(data + i), v);
            }

            // оставшиеся
            for (; i < count; ++i) {
                uint16_t val = data[i];
                data[i] = (val >> 8) | (val << 8);
            }
        }

        void load(const std::string& filename) {
            std::ifstream in(filename, std::ios::binary);
            if (!in) {
                throw std::runtime_error("Failed to open file for reading");
            }
            if (get_next_line(in) != "P6")
                throw std::invalid_argument("Invalid PGM format\nuse: ffmpeg -i input.png -pix_fmt rgb48 output.ppm");

            std::istringstream pgm_width_height_is(get_next_line(in,true));
            std::istringstream pgm_bits_per_channel_is(get_next_line(in));

            pgm_width_height_is >> width >> height;
            uint32_t maxvalue{};
            pgm_bits_per_channel_is >> maxvalue;
            bits_per_channel = 32 - lzcnt_s( maxvalue );
            channel_nb = 3;

            if (width == 0 || height == 0 || bits_per_channel == 0 || bits_per_channel > 16 )
                throw std::invalid_argument("Invalid PPM format");
            auto size = size_t(width) * height * channel_nb * (bits_per_channel > 8 ? 2 : 1);
            data = std::make_unique<uint8_t[]>(size);
            in.read(reinterpret_cast<char*>(data.get()),size);
            if (bits_per_channel > 8) {
                channel_type = ChannelType::uint16be;
            } else {
                channel_type = ChannelType::uint8;
            }

        }
        inline void save( const std::string& filename ) {
            using ChannelType = llcomp::RawImage::ChannelType;

            if (channel_nb != 3) {
                throw std::runtime_error("channel_nb != 3");
            }
            if (!(channel_type == ChannelType::uint8 || channel_type == ChannelType::uint16 || channel_type == ChannelType::uint16be )) {
                throw std::runtime_error("Unsupported channel type for PPM output");
            }

            std::ofstream out(filename, std::ios::binary);
            if (!out) {
                throw std::runtime_error("Failed to open file for writing");
            }

            out << "P6\n" << width << ' ' << height << "\n" << (uint32_t{1} << bits_per_channel)-1 << "\n";

            be();
            out.write(reinterpret_cast<char*>(data.get()), width*height*3* (channel_type==ChannelType::uint8 ? 1 : 2));

        }
        inline void le() {
            if (channel_type == ChannelType::uint16be) {
                swap_bytes_16bit_sse(as<uint16_t>(), width * height * channel_nb);
                channel_type = ChannelType::uint16;
            }
        }
        inline void be() {
            if (channel_type == ChannelType::uint16) {
                swap_bytes_16bit_sse(as<uint16_t>(), width * height * channel_nb);
                channel_type = ChannelType::uint16be;
            }
        }

    };

    template <typename T>
    inline void decodeChannel(uint32_t width, uint32_t height, const uint8_t *data_begin, const uint8_t *data_end, T *pixels)
    {
        size_t rgb_pos = 0;
        auto data = data_begin;

        BitStream::Reader bit_stream{[&]()
                                     {
                                         if (data + 8 <= data_end)
                                         {
                                             data += 8;
                                             return std::make_pair(data - 8, size_t{8});
                                         }
                                         else
                                         {
                                             size_t rem = data_end - data;
                                             return std::make_pair(data_end - rem, rem);
                                         }
                                     }};
        bit_stream.init();

        RLGR::Decoder decoder{[&]
                              { return bit_stream.peek32(); }, [&](uint32_t n)
                              { return bit_stream.skip(n); }};

        auto [next, ziros] = decoder.decode();
        for (uint32_t h = 0; h < height; ++h)
        {
            for (uint32_t w = 0; w < width; ++w)
            {
                const int l = w > 0 ? pixels[-1] : (h > 0 ? pixels[-static_cast<ptrdiff_t>(width)] : 0);
                const int t = h > 0 ? pixels[-static_cast<ptrdiff_t>(width)] : l;
                const int tl = h > 0 && w > 0 ? pixels[-static_cast<ptrdiff_t>(width) - 1] : t;
                const int predict = median(l, l + t - tl, t);

                int pixel = predict + (ziros == 0 ? next : 0);
                if constexpr (std::is_same_v<T, int32_t> == false ) {
                    if (pixel < std::numeric_limits<T>::min() || pixel > std::numeric_limits<T>::max()) {
                        throw std::runtime_error( "Pixel value out of range");
                    }
                }
                *pixels++ = pixel;

                if (ziros == 0)
                    std::tie(next, ziros) = decoder.decode();
                else
                    ziros--;
            }
        }
    }

    struct Header
    {
        uint32_t hdr_crc;
        uint32_t width;         // same for all channels
        uint32_t height;        // same for all channels
        uint16_t flags;          // 15: is rgb (first 3 channels is red, green, blue channels
        uint8_t  channel_nb;    // r,g,b,n1,n2,n3,n4...
        uint8_t  channels_depth; // 8-16 supported: 8-15 -> int16_t arithmetic coding, 16 -> int32_t arithmetic coding, >16 is not tested
        void protect() {
            hdr_crc = _mm_crc32_u32(signature,width);
            hdr_crc = _mm_crc32_u32(hdr_crc, height);
            hdr_crc = _mm_crc32_u16(hdr_crc, flags);
            hdr_crc = _mm_crc32_u8(hdr_crc, channel_nb);
            hdr_crc = _mm_crc32_u8(hdr_crc, channels_depth);
        }
        bool check()  {
            auto crc = _mm_crc32_u32(signature,width);
            crc = _mm_crc32_u32(crc, height);
            crc = _mm_crc32_u16(crc, flags);
            crc = _mm_crc32_u8(crc, channel_nb);
            crc = _mm_crc32_u8(crc, channels_depth);
            return crc == hdr_crc;
        }
    };

    template <int src_depth, int dest_depth, typename T>
    std::vector<uint8_t> compressImage(const T *rgb, uint32_t width, uint32_t height, uint8_t channels)
    {
        auto planar = split_rgb_into_planar<src_depth, dest_depth>(rgb, channels, width, height);
        auto out = std::vector(planar.size(), std::vector(0, uint8_t{}));
        constexpr uint8_t channels_depth = dest_depth;
        std::vector<std::future<std::vector<uint8_t>>> futures;
        futures.reserve(channels);

        for (size_t i = 0; i < out.size(); ++i)
        {
            futures.push_back(
                std::async(i==0 ? std::launch::deferred : std::launch::async, [&](size_t n)
                           { return encodeChannel(planar[n].data(), width, height); }, i));
        }

        Header hdr{0, width, height, uint16_t(1 << 15), channels, channels_depth };
        if ( std::is_same_v<T, uint8_t> && channels_depth > 8 ) {
            throw std::invalid_argument("std::is_same_v<T, uint8_t> && channels_depth > 8");
        }
        if ( std::is_same_v<T, uint16_t> && channels_depth <= 8) {
            throw std::invalid_argument("std::is_same_v<T, uint16_t> && channels_depth <= 8");
        }
        if ( std::is_same_v<T, uint32_t> && channels_depth <= 16) {
            throw std::invalid_argument("std::is_same_v<T, uint32_t> && channels_depth <= 16");
        }
        hdr.protect();
        std::vector<uint8_t> hdr_and_channels;
        hdr_and_channels.reserve(sizeof(hdr) + width * height * channels * 3 / 4);
        auto bi = std::back_inserter(hdr_and_channels);
        std::copy(reinterpret_cast<const uint8_t *>(&hdr), reinterpret_cast<const uint8_t *>(&hdr) + sizeof(hdr), bi);

        for (size_t i = 0; i < futures.size(); ++i)
        {
            auto vec = futures[i].get();
            size_t vec_size = vec.size();
            std::copy(reinterpret_cast<const uint8_t *>(&vec_size), reinterpret_cast<const uint8_t *>(&vec_size) + sizeof(vec_size), bi);
            std::copy(vec.begin(), vec.end(), bi);
        }
        return hdr_and_channels;
    }


    template <typename DestType>
    auto decompressImageAfterHeaderAsPlanar(const Header &hdr, const uint8_t *begin, const uint8_t *end)
    {
        std::vector<std::vector<DestType>> planar(hdr.channel_nb, std::vector<DestType>(hdr.width * hdr.height));
        std::vector<std::future<bool>> futures;
        futures.reserve(hdr.channel_nb);
        for (size_t i = 0; i < planar.size(); ++i)
        {
            size_t size;
            std::copy(begin, begin + sizeof(size), reinterpret_cast<uint8_t *>(&size));
            begin += sizeof(size);
            futures.push_back(
                std::async( i==0 ? std::launch::deferred : std::launch::async, [&](const uint8_t *begin, const uint8_t *end, DestType *pixels)
                           {
                               decodeChannel(hdr.width, hdr.height, begin, end, pixels);
                               return true;
                           },
                           begin, begin + size, planar[i].data()));
            begin += size;
        };
        for (auto &future : futures)
            future.get();

        return planar;

    }



    RawImage decompressImage(const uint8_t *begin, const uint8_t *end)
    {
        Header hdr{};
        std::copy(begin, begin + sizeof(hdr), reinterpret_cast<uint8_t *>(&hdr));
        begin += sizeof(hdr);
        if (!hdr.check()) {
            throw std::runtime_error("invalid format");
        }

        if (hdr.channels_depth <= 8) {
            auto planar = decompressImageAfterHeaderAsPlanar<int16_t>(hdr,begin,end);
            RawImage rawImage{
                std::make_unique<uint8_t[]>(hdr.width * hdr.height * hdr.channel_nb),
                hdr.width, hdr.height, hdr.channel_nb, hdr.channels_depth, RawImage::ChannelType::uint8};

            join_planar_into_rgb(planar,hdr.width,hdr.height,rawImage.as<uint8_t>());
            return rawImage;
        } else
        if (hdr.channels_depth < 16) {
            auto planar = decompressImageAfterHeaderAsPlanar<int16_t>(hdr,begin,end);
            RawImage rawImage{
                std::make_unique<uint8_t[]>(hdr.width * hdr.height * hdr.channel_nb * sizeof(uint16_t)),
                hdr.width, hdr.height, hdr.channel_nb, hdr.channels_depth, RawImage::ChannelType::uint16};

            join_planar_into_rgb(planar,hdr.width,hdr.height,rawImage.as<uint16_t>());
            return rawImage;
        } else
        if (hdr.channels_depth == 16) {
            auto planar = decompressImageAfterHeaderAsPlanar<int32_t>(hdr,begin,end);
            RawImage rawImage{
                std::make_unique<uint8_t[]>(hdr.width * hdr.height * hdr.channel_nb * sizeof(uint16_t)),
                hdr.width, hdr.height, hdr.channel_nb, hdr.channels_depth, RawImage::ChannelType::uint16};
            join_planar_into_rgb(planar,hdr.width,hdr.height,rawImage.as<uint16_t>());
            return rawImage;
        } else {
            auto planar = decompressImageAfterHeaderAsPlanar<int32_t>(hdr,begin,end);
            RawImage rawImage{
                std::make_unique<uint8_t[]>(hdr.width * hdr.height * hdr.channel_nb * sizeof(int32_t)),
                hdr.width, hdr.height, hdr.channel_nb, hdr.channels_depth, RawImage::ChannelType::int32};
            join_planar_into_rgb(planar,hdr.width,hdr.height,rawImage.as<int32_t>());
            return rawImage;
        }
    }

}