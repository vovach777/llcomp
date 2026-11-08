#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include <stdexcept>
#include "llcomp.hpp"
#include "netpbm.hpp"


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_file>" << std::endl;
        return 1;
    }
    try {
        Netpbm ppm(argv[1]);
        if (ppm.type != '6' && ppm.type != '5') {
            throw std::invalid_argument("Unsupported Netpbm type (only P6/P5 are supported)");
        }
        if (ppm.maxvalue != 255) {
            throw std::invalid_argument("Unsupported maxvalue (only 8-bit, 255, is supported)");
        }
        if (ppm.width == 0 || ppm.height == 0 || ppm.width > 65535 || ppm.height > 65535) {
            throw std::invalid_argument("Invalid image dimensions");
        }
        const uint32_t width = ppm.width;
        const uint32_t height = ppm.height;
        const uint32_t channels = (ppm.type == '6') ? 3 : 1;
        std::vector<uint8_t> rgb(size_t(width) * height * channels);
        ppm.read(rgb.data(), rgb.size());
        ppm.close();

        std::vector<uint8_t> compressed = llcomp::compressImage(rgb, width, height, channels);
        const char* outputFile = argv[2];
        std::ofstream outFile(outputFile, std::ios::binary);
        if (!outFile) {
            throw std::runtime_error(std::string("Error opening output file: ") + outputFile);
        }
        outFile.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
        outFile.close();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error compressing image: " << e.what() << std::endl;
        return 1;
    }
}
