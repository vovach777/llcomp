#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include "llcomp.hpp"

void write_ppm_p6(const std::string& filename, const llcomp::RawImage& raw) {
    using ChannelType = llcomp::RawImage::ChannelType;

    if (raw.channel_nb < 3) {
        throw std::runtime_error("Need at least 3 channels for RGB output");
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing");
    }

    const int width  = raw.width;
    const int height = raw.height;

    if (raw.channel_type == ChannelType::uint16) {
        out << "P6\n" << width << ' ' << height << "\n65535\n";

        const uint16_t* data = raw.as<uint16_t>();
        for (int i = 0; i < width * height; ++i) {
            const uint16_t* pixel = data + i * raw.channel_nb;
            for (int c = 0; c < 3; ++c) {
                uint16_t val = pixel[c];
                char bytes[2] = {
                    static_cast<char>(val >> 8),       // MSB
                    static_cast<char>(val & 0xFF)      // LSB
                };
                out.write(bytes, 2);
            }
        }

    } else if (raw.channel_type == ChannelType::uint8) {
        out << "P6\n" << width << ' ' << height << "\n255\n";

        const uint8_t* data = raw.as<uint8_t>();
        for (int i = 0; i < width * height; ++i) {
            const uint8_t* pixel = data + i * raw.channel_nb;
            out.write(reinterpret_cast<const char*>(pixel), 3);  // Только RGB
        }

    } else {
        throw std::runtime_error("Unsupported channel type for PPM output");
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    const char* filename = argv[1];
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening input file: " << filename << std::endl;
        return 1;
    }
    std::vector<uint8_t> compressed((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();
    try {
        auto rawImage = llcomp::decompressImage(compressed.data(), compressed.data()+compressed.size() );
        std::string outputFile = std::string(filename) + ".pnm";
        write_ppm_p6(outputFile, rawImage);

    } catch (const std::exception& e) {
        std::cerr << "Error decompressing image: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }

    return 0;
}
