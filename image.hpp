#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include <nmmintrin.h>
#include <string>
#include <cstring>
#include <string_view>
#include <sstream>
#include <fstream>


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
            using ChannelType = RawImage::ChannelType;

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
