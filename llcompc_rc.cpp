#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include <thread>
#include <algorithm>
#include <stdexcept>
#include "llcomp_rc.hpp"
#include "netpbm.hpp"


int main(int argc, char** argv) {
    // [-t T] threads. Default 1: the whole frame is one piece. T>1 cuts T row-bands, but not under 512 KiB.
    int mode = 0, threads = 1; std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-t" && i + 1 < argc) threads = std::max(1, std::stoi(argv[++i]));
        else if (a == "-x" && i + 1 < argc) mode = std::stoi(argv[++i]);
        else pos.push_back(argv[i]);
    }
    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " [-t threads] [-x 0..11] <input_image> <output_file>\n"
                     "  -t T  T row-bands, one thread each (default 1, the whole frame). A band stays >= 512 KiB, else T is lowered\n"
                     "  -x M  context mode 0..11" << std::endl;
        return 1;
    }
    try {
        Netpbm ppm(pos[0]);
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

        std::vector<uint8_t> compressed = llcomp_rc::compressImage(rgb, width, height, channels, threads, 0, mode);
        const char* outputFile = pos[1];
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
