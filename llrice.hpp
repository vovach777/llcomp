#pragma once
#include <functional>
#include <cassert>
#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <type_traits>
#include <limits>
#include <type_traits>
#include <tuple>
#include "pool.hpp"
#include "bitstream.hpp"
#include "rlgr.hpp"
#include "image.hpp"

//#define USE_SIMPLE_RLGR
#define ONE_CODER
//#define USE_RLGR3
#define USE_NEW_LINE
namespace llrice
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
            Encoder g_encoder(out_pool);
            Encoder rb_encoder(out_pool);
        #ifdef USE_RLGR3
            BitStream::Writer r_encoder(out_pool);
        #endif
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
                        g_encoder.put( v1 ); //g is luma
                        #ifdef USE_RLGR3
                            auto rb = v0 + v2;    //rb is chroma
                            rb_encoder.put( rb );
                            if (unlikely(rb != 0)) {
                                auto k_rem = std20::bit_width( rb );
                                r_encoder.reserve( k_rem );
                                r_encoder.put_bits( k_rem, v0 );
                            }
                        #else
                            rb_encoder.put( v0 );
                            rb_encoder.put( v2 );
                        #endif

                    #else
                        encoders[0].put( v0 );
                        encoders[1].put( v1 );
                        encoders[2].put( v2 );
                    #endif

            }
        #ifdef USE_NEW_LINE
            #ifdef ONE_CODER
                g_encoder.new_line();
                rb_encoder.new_line();
            #else
                encoders[0].new_line();
                encoders[1].new_line();
                encoders[2].new_line();
            #endif
        #endif

        }
#ifdef ONE_CODER
        g_encoder.flush();
        rb_encoder.flush();
    #ifdef USE_RLGR3
        r_encoder.flush();
    #endif
#else
        for (int i = 0; i < 3; ++i) {
            encoders[i].flush();
        }
#endif
        return out_pool.move();
    }



    inline auto rlgr_decode(std::vector<uint64_t> & poolvec)
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
        Decoder g_decoder(pool);
        Decoder rb_decoder(pool);
    #ifdef USE_RLGR3
        BitStream::Reader r_decoder(pool);
    #endif
#else
        std::array<Decoder,3> decoders{
            Decoder(pool),
            Decoder(pool),
            Decoder(pool)
        };
#endif
        int clamp = (1 << img.bits_per_channel)-1;
        auto core = [&](auto store) {
        for (uint32_t h = 0; h < height; ++h) {
            auto curr_line = line[h & 1];
            auto prev_line = line[(h-1) & 1];
            for (uint32_t w = 0; w < width*4; w+=4) {
                #ifdef ONE_CODER
                    int16_t v0{0};
                    int16_t v1 = Rice::to_signed( g_decoder.get() );
                    int16_t v2{0};
                    #ifdef USE_RLGR3
                        auto rb = rb_decoder.get();
                        if (unlikely(rb != 0)) {
                            auto r_rem = std20::bit_width(rb);
                            r_decoder.reserve(r_rem);
                            auto r = r_decoder.peek_n(r_rem);
                            r_decoder.skip(r_rem);
                            v0 = Rice::to_signed( r );
                            v2 = Rice::to_signed( rb - r );
                        }
                    #else
                        v0 = Rice::to_signed( rb_decoder.get() );
                        v2 = Rice::to_signed( rb_decoder.get() );

                    #endif

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
        #ifdef USE_NEW_LINE
            #ifdef ONE_CODER
                g_decoder.new_line();
                rb_decoder.new_line();
            #else
                decoders[0].new_line();
                decoders[1].new_line();
                decoders[2].new_line();
            #endif
        #endif
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

    inline RawImage decompressImage(std::vector<uint64_t>& poolvec)
    {
        return rlgr_decode(poolvec);
    }

}