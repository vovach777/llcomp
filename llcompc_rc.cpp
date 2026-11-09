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
    // [-s 1|2|4|8] band size in Ivy Bridge L2 units (256 KiB of pixels, whole rows). [-t T] worker threads.
    int mode = 0, band = 1, threads = int(std::thread::hardware_concurrency()); std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-s" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v != "1" && v != "2" && v != "4" && v != "8") {
                std::cerr << "band mode must be 1, 2, 4 or 8\n";
                return 1;
            }
            band = std::stoi(v);
        }
        else if (a == "-t" && i + 1 < argc) threads = std::max(1, std::stoi(argv[++i]));
        else if (a == "-x" && i + 1 < argc) mode = std::stoi(argv[++i]);
        else pos.push_back(argv[i]);
    }
    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " [-s 1|2|4|8] [-t threads] [-x 0..11] <input_image> <output_file>\n"
                     "  -s K  one piece is K * 256 KiB of pixels, cut on a row (default 1). Not a column strip\n"
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

        std::vector<uint8_t> compressed = llcomp_rc::compressImage(rgb, width, height, channels, threads, band, mode);
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
