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
#include <cstring>
#include <string_view>
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
#include <tuple>
#include "pool.hpp"
#include "bitstream.hpp"
#include "rlgr.hpp"
//#define USE_SIMPLE_RLGR
//#define ONE_CODER
namespace llcomp
{
    constexpr inline auto ext = ".llr";
    constexpr inline uint8_t rev = '1';
    constexpr inline uint32_t signature = 'l' | 'l' << 8U | 'r' << 16U | rev << 24U;

    template <typename T>
    inline int median(T a, T b, T c)
    {
        //return a + b + c - std::max({a, b, c}) - std::min({a, b, c});
        return std::clamp(c, std::min(a, b), std::max(a, b));
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


    template <typename T>
    inline auto predict(uint32_t w, uint32_t h, const T* curr_row, const T* prev_row) {
        const T l = w >= 4 ? curr_row[w - 4] : h > 0 ? prev_row[w] : 0;
        const T t = h > 0 ? prev_row[w] : l;
        const T tl = h > 0 && w >= 4 ? prev_row[w-4] : t;
        return median<T>(l, l + t - tl, t);
    }

    template <typename RGBLoader>
    auto rlgr_encode(const Header& hdr, RGBLoader && rgbloader)
    {
        const auto width = hdr.width;
        const auto height = hdr.height;
        std::vector<int16_t> lines( 2*4*width );
        int16_t* line[2] {
            lines.data(),
            lines.data() + width*4
        };
        PagePool out_pool;
        out_pool.reserve(hdr.width*hdr.height);
        out_pool.acquire_page(); //для заголовка
        out_pool.acquire_page();
        std::memcpy(&(out_pool[0]), &hdr, sizeof(Header));

#ifdef USE_SIMPLE_RLGR
        using Encoder = RLGR::Simple::Encoder;
#else
        using Encoder = RLGR::Encoder;
#endif

    #ifdef ONE_CODER
            Encoder encoder(out_pool);
    #else
        std::array<Encoder,3> encoders{
            Encoder(out_pool),
            Encoder(out_pool),
            Encoder(out_pool)
        };
    #endif

        // uint16_t Rshift = std::max(0, hdr.channels_depth - 8);
        // uint16_t Rmask = (1 << Rshift) - 1;


        for (uint32_t h = 0; h < height; ++h) {
            auto curr_line = line[h & 1];
            auto prev_line = line[(h-1) & 1];
            for (uint32_t w = 0; w < width*4; w+=4) {
                    auto [r,g,b] = rgbloader();
                    b -= g;
                    r -= g;
                    g += (b + r) >> 2;

                    curr_line[w+0] = r;
                    curr_line[w+1] = g;
                    curr_line[w+2] = b;
                    auto v0 = Rice::to_unsigned( r - predict(w+0,h,curr_line, prev_line) );
                    auto v1 = Rice::to_unsigned( g - predict(w+1,h,curr_line, prev_line) );
                    auto v2 = Rice::to_unsigned( b - predict(w+2,h,curr_line, prev_line) );
                    #ifdef ONE_CODER
                        encoder.put( v0 );
                        encoder.put( v1 );
                        encoder.put( v2 );
                    #else
                        encoders[0].put( v0 );
                        encoders[1].put( v1 );
                        encoders[2].put( v2 );
                    #endif

            }
        }
#ifdef ONE_CODER
        encoder.flush();
#else
        for (int i = 0; i < 3; ++i) {
            encoders[i].flush();
        }
#endif
        return out_pool.move();
    }


    struct RawImage
    {
        private:
        std::unique_ptr<uint16_t[]> data{};
        public:
        uint32_t width{};
        uint32_t height{};
        uint8_t channel_nb{};
        uint8_t bits_per_channel{};
        enum class ChannelType{ none, uint8, uint16, uint16be };
        ChannelType channel_type{ ChannelType::none };

        void allocate() {
            if (channel_type != ChannelType::none) {
                size_t size = width * height * channel_nb;

                if ( channel_type == ChannelType::uint8 )
                    size = (size + 1) / 2;
                data = std::make_unique<uint16_t[]>( size );
            }
        }

        template<typename T>
        T* as() {
            static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>,
                "Unsupported type in RawImage::as()");
            return reinterpret_cast<T*>(data.get());
        }
        template<typename T>
        const T* as() const {
            static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>,
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
                throw std::invalid_argument("Invalid PPM format\nuse: ffmpeg -i input.png -pix_fmt rgb48 output.ppm");

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
            if (bits_per_channel > 8) {
                channel_type = ChannelType::uint16be;
            } else {
                channel_type = ChannelType::uint8;
            }
            allocate();
            in.read(reinterpret_cast<char*>(data.get()),size);
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

    inline auto rlgr_decode(std::vector<Chunk32> & poolvec)
    {
        size_t rgb_pos = 0;
        Header hdr;
        RawImage img;

        PagePool pool(std::move(poolvec));
        pool.get_next_read_page(); //пропускаем заголовок
        pool.get_next_read_page();

        std::memcpy(&hdr, &(pool[0]), sizeof(Header));
        if (!hdr.check()) {
            throw std::runtime_error("Header CRC check failed");
        }
        img.width = hdr.width;
        img.height = hdr.height;
        img.channel_nb = 3;
        img.bits_per_channel = hdr.channels_depth;
        //RawImage::ChannelType::uint16be  pnm freadly
        img.channel_type = hdr.channels_depth <= 8 ? RawImage::ChannelType::uint8 : RawImage::ChannelType::uint16be;
        img.allocate();
        auto width = img.width;
        auto height = img.height;

        std::vector<int16_t> lines( 2*4*width );
        int16_t* line[2] {
            lines.data(),
            lines.data() + width*4
        };

        size_t count = size_t(width) * height * 3;
        using Stream = BitStream::Reader;
#ifdef USE_SIMPLE_RLGR
        using Decoder = RLGR::Simple::Decoder;
#else
        using Decoder = RLGR::Decoder;
#endif

#ifdef ONE_CODER
        Decoder decoder(pool ,count);
#else
        std::array<Decoder,3> decoders{
            Decoder(pool, (count / 3) + (0 < (count % 3))),
            Decoder(pool, (count / 3) + (1 < (count % 3))),
            Decoder(pool, (count / 3) + (2 < (count % 3)))
        };
#endif
        int clamp = (1 << img.bits_per_channel)-1;
        auto core = [&](auto store) {
        for (uint32_t h = 0; h < height; ++h) {
            auto curr_line = line[h & 1];
            auto prev_line = line[(h-1) & 1];
            for (uint32_t w = 0; w < width*4; w+=4) {
                #ifdef ONE_CODER
                    int16_t v0 = Rice::to_signed( decoder.get() );
                    int16_t v1 = Rice::to_signed( decoder.get() );
                    int16_t v2 = Rice::to_signed( decoder.get() );
                #else
                    int16_t v0 = Rice::to_signed( decoders[0].get() );
                    int16_t v1 = Rice::to_signed( decoders[1].get() );
                    int16_t v2 = Rice::to_signed( decoders[2].get() );
                #endif
                v0 += predict(w + 0,h,curr_line,prev_line);
                v1 += predict(w + 1,h,curr_line,prev_line);
                v2 += predict(w + 2,h,curr_line,prev_line);
                auto r = curr_line[w+0] = v0;
                auto g = curr_line[w+1] = v1;
                auto b = curr_line[w+2] = v2;
                g -= (r + b) >> 2;
                r += g;
                b += g;
                store(r,g,b);
            }
        }
        };//lambda
        if (img.channel_type == RawImage::ChannelType::uint16) {
            uint16_t* pixels16 = img.as<uint16_t>();
            core([&](int16_t r, int16_t g, int16_t b) {
                *pixels16++=r;
                *pixels16++=g;
                *pixels16++=b;
            });
        } else
        if (img.channel_type == RawImage::ChannelType::uint16be) {
            uint16_t* pixels16 = img.as<uint16_t>();
            core([&](int16_t r, int16_t g, int16_t b) {
                *pixels16++=bswap16(r);
                *pixels16++=bswap16(g);
                *pixels16++=bswap16(b);
            });
        } else {
            assert( img.channel_type == RawImage::ChannelType::uint8 );
            uint8_t* pixels8 = img.as<uint8_t>();
            core([&](int8_t r, int8_t g, int8_t b) {
                *pixels8++=r;
                *pixels8++=g;
                *pixels8++=b;
            });
        }
        return img;
    }



    inline auto compressImage(const RawImage& img)
    {
        Header hdr{0, img.width, img.height,static_cast<uint16_t>(1 << 15), img.channel_nb, img.bits_per_channel };
        hdr.protect();
        if ( img.channel_type == RawImage::ChannelType::uint16 ) {
            auto pixels = img.as<uint16_t>();
            return rlgr_encode(hdr, [&]()
            {
                pixels += 3;
                return std::make_tuple( int(pixels[-3]), int(pixels[-2]), int(pixels[-1]));
            });
        } else
        if ( img.channel_type == RawImage::ChannelType::uint8 ) {
            auto pixels = img.as<uint8_t>();
            return rlgr_encode(hdr, [&]()
            {
                pixels += 3;
                return std::make_tuple( int16_t(pixels[-3]), int16_t(pixels[-2]), int16_t(pixels[-1]));
            });
        }
        throw std::runtime_error("Unsupported channel type");


    }

    inline RawImage decompressImage(std::vector<Chunk32>& poolvec)
    {
        return rlgr_decode(poolvec);
    }

}