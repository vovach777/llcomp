#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include <string>
#include <cstring>
#include <string_view>
#include <sstream>
#include <fstream>
#include <vector>

    struct Netpbm
    {
        char type{'E'};
        uint32_t maxvalue{};
        uint32_t width{};
        uint32_t height{};
        std::ifstream in;
        std::ofstream out;

        static uint32_t read_next_int(std::ifstream& stream) {
            char ch;
            while (stream >> std::ws) { // Пропускаем пробельные символы
                ch = stream.peek();
                if (ch == '#') {
                    std::string dummy;
                    std::getline(stream, dummy); // Скидываем комментарий до конца строки
                } else {
                    uint32_t value;
                    if (stream >> value) {
                        return value;
                    }
                    break;
                }
            }
            throw std::runtime_error("Invalid PPM header structure");
        }

        Netpbm(const std::string filename) : in(filename, std::ios::binary) {
            in.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
            std::string type_str;
            in >> type_str; // Читаем первый токен (например, P6)

            if (type_str.size() != 2 || type_str[0] != 'P')
                throw std::invalid_argument("Invalid PPM format\nuse: ffmpeg -i input.png -pix_fmt rgb24 output.ppm");
            type = type_str[1];
            width = read_next_int(in);
            height = read_next_int(in);
            maxvalue = read_next_int(in);
            // Спецификация Netpbm: строго ОДИН пробельный символ после maxvalue перед бинарными данными
            char dummy_space;
            in.get(dummy_space);
        }

        Netpbm( const std::string filename, char type, uint32_t width, uint32_t height, uint32_t maxvalue ) : out(filename, std::ios::binary), type(type), width(width), height(height),maxvalue(maxvalue) {
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out << 'P' << type << '\n' << width << ' ' << height << '\n' << maxvalue << '\n';
        }
        void write(const void * data, size_t size) {
            if (data == nullptr) {
                out.close();
                out.exceptions(std::ios::goodbit);
                return;
            }
            if (size == 0) {
                return;
            }
            out.write(reinterpret_cast<const char*>(data), size);
        }
        void read(void * data, size_t size) {
            if (data == nullptr) {
                in.close();
                in.exceptions(std::ios::goodbit);
                return;
            }
            if (size == 0) {
                return;
            }
            in.read(reinterpret_cast<char*>(data), size);
            // if (static_cast<size_t>( in.gcount() ) < size ) {
            //     throw std::runtime_error("PPM data truncated");
            // }
        }

        void close() {
            out.close();
            out.exceptions(std::ios::goodbit);
            in.close();
            in.exceptions(std::ios::goodbit);
        }

        ~Netpbm() {
            out.exceptions(std::ios::goodbit);
            in.exceptions(std::ios::goodbit);
        }
    };
