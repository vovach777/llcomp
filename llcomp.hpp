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
#include "pool.hpp"
#include "bitstream.hpp"
#include "rlgr.hpp"
//#define USE_SIMPLE_RLGR

namespace llcomp
{
    constexpr inline auto ext = ".llr";
    constexpr inline uint8_t rev = '1';
    constexpr inline uint32_t signature = 'l' | 'l' << 8U | 'r' << 16U | rev << 24U;

    inline int median(int a, int b, int c)
    {
        return a + b + c - std::max({a, b, c}) - std::min({a, b, c});
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


    template <typename RGBLoader>
    auto rlgr_encode(const Header& hdr, RGBLoader && rgbloader)
    {
        const auto width = hdr.width;
        const auto height = hdr.height;
        //std::vector<std::vector<std::vector<int>>> line(2, std::vector<std::vector<int>>(3, std::vector<int>(width, 0)));
        std::vector<int> lines = std::vector<int>(2*3*width);
        int* line[2][3] = {
                            { lines.data(),         lines.data()+width,    lines.data()+width*2},
                            { lines.data()+width*3, lines.data()+width*4,  lines.data()+width*5}
                        };

        PagePool out_pool;
        out_pool.reserve(hdr.width*hdr.height);
        out_pool.acquire_page(); //для заголовка
        out_pool.acquire_page();

        std::memcpy(&(out_pool[0]), &hdr, sizeof(Header));
#ifdef USE_SIMPLE_RLGR
        std::array<RLGR::Simple::Encoder,3> encoders{
            RLGR::Simple::Encoder(out_pool),
            RLGR::Simple::Encoder(out_pool),
            RLGR::Simple::Encoder(out_pool)
        };
#else
        std::array<RLGR::Encoder,3> encoders{
            RLGR::Encoder(out_pool),
            RLGR::Encoder(out_pool),
            RLGR::Encoder(out_pool)
        };
#endif

        // uint16_t Rshift = std::max(0, hdr.channels_depth - 8);
        // uint16_t Rmask = (1 << Rshift) - 1;


        for (uint32_t h = 0; h < height; ++h) {
            for (uint32_t w = 0; w < width; ++w) {
                    auto urgb = rgbloader();
                    auto& r = line[h & 1][0][w];
                    auto& g = line[h & 1][1][w];
                    auto& b = line[h & 1][2][w];

                    r = urgb[0];
                    g = urgb[1];
                    b = urgb[2];

                    b -= g;
                    r -= g;
                    g += (b + r) / 4;
                    #pragma unroll(3)
                    for (int i = 0; i < 3; ++i) {
                        const int l = w > 0 ? line[h & 1][i][w - 1] : h > 0 ? line[(h - 1) & 1][i][w] : 0;
                        const int t = h > 0 ? line[(h-1)&1][i][w] : l;
                        const int tl = h > 0 && w > 0 ? line[(h-1)&1][i][w-1] : t;
                        const int predict = median(l, l + t - tl, t);
                        #ifdef ONE_CODER
                           encoder.put( Rice::to_unsigned( line[h&1][i][w] - predict));
                        #else
                        encoders[i].put( Rice::to_unsigned( line[h&1][i][w] - predict));
                        #endif

                    }
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

    inline auto rlgr_decode(std::vector<uint64_t> & poolvec)
    {
        size_t rgb_pos = 0;
        Header hdr;
        PagePool pool(std::move(poolvec));
        auto hdri = pool.get_next_read_page(); pool.get_next_read_page();

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
        img.channel_type = hdr.channels_depth <= 8 ? RawImage::ChannelType::uint8 : RawImage::ChannelType::uint16;
        img.allocate();
        auto width = img.width;
        auto height = img.height;
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
        std::array<RLGR::Decoder,3> decoders{
            Decoder(pool, (count / 3) + (0 < (count % 3))),
            Decoder(pool, (count / 3) + (1 < (count % 3))),
            Decoder(pool, (count / 3) + (2 < (count % 3)))
        };
#endif
        uint8_t* pixels = img.as<uint8_t>();
        uint16_t* pixels16 = img.as<uint16_t>();
        bool is_16bit = img.bits_per_channel > 8;
        int clamp = (1 << img.bits_per_channel)-1;
        //std::vector<std::vector<std::vector<int>>> line(2, std::vector<std::vector<int>>(3, std::vector<int>(width, 0)));
        std::vector<int> lines = std::vector<int>(2*3*width);
        int* line[2][3] = {
                            { lines.data(),         lines.data()+width,    lines.data()+width*2},
                            { lines.data()+width*3, lines.data()+width*4,  lines.data()+width*5}
                        };

        for (uint32_t h = 0; h < height; ++h)
        {
            for (uint32_t w = 0; w < width; ++w)
            {
                for (int i = 0; i < 3; ++i) {
                    const int l = w > 0 ? line[h & 1][i][w - 1] : h > 0 ? line[(h - 1) & 1][i][w] : 0;
                    const int t = h > 0 ? line[(h-1)&1][i][w] : l;
                    const int tl = h > 0 && w > 0 ? line[(h-1)&1][i][w-1] : t;
                    #ifdef ONE_CODER
                    line[h & 1][i][w] = Rice::to_signed( decoder.get() ) + median(l, l + t - tl, t);
                    #else
                    line[h & 1][i][w] = Rice::to_signed( decoders[i].get() ) + median(l, l + t - tl, t);
                    #endif
                }
                int r = line[h & 1][0][w];
                int g = line[h & 1][1][w];
                int b = line[h & 1][2][w];
                g -= ((r + b) / 4);
                r += g;
                b += g;
                if (is_16bit) {
                    pixels16[0] = r;
                    pixels16[1] = g;
                    pixels16[2] = b;
                    pixels16 += 3;
                } else {
                    pixels[0] = r;
                    pixels[1] = g;
                    pixels[2] = b;
                    pixels += 3;
                }
            }
        }
        return img;
    }



    auto compressImage(const RawImage& img)
    {
        Header hdr{0, img.width, img.height,static_cast<uint16_t>(1 << 15), img.channel_nb, img.bits_per_channel };
        hdr.protect();
        if ( img.channel_type == RawImage::ChannelType::uint16 ) {
            auto pixels = img.as<uint16_t>();
            return rlgr_encode(hdr, [&]()
            {
                pixels += 3;
                return std::array<std::uint16_t,3>{ pixels[-3], pixels[-2], pixels[-1] };
            });
        } else
        if ( img.channel_type == RawImage::ChannelType::uint8 ) {
            auto pixels = img.as<uint8_t>();
            return rlgr_encode(hdr, [&]()
            {
                pixels += 3;
                return std::array<std::uint16_t,3>{ uint16_t(pixels[-3]), uint16_t(pixels[-2]), uint16_t(pixels[-1]) };
            });
        }
        throw std::runtime_error("Unsupported channel type");


    }

    RawImage decompressImage(std::vector<uint64_t>& poolvec)
    {
        return rlgr_decode(poolvec);
    }

}