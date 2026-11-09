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
    int threads = 1; std::vector<const char*> pos;   // [-t T] worker threads, default 1
    for (int i = 1; i < argc; ++i) { std::string a = argv[i]; if (a == "-t" && i + 1 < argc) threads = std::max(1, std::stoi(argv[++i])); else pos.push_back(argv[i]); }
    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " [-t threads] <input_file> <output_image>" << std::endl;
        return 1;
    }
    std::ifstream inFile(pos[0], std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening input file: " << pos[0] << std::endl;
        return 1;
    }
    std::vector<uint8_t> compressed((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();

    try {
        auto [pixels, width, height, channels] = llcomp_rc::decompressImage(compressed, threads);

        const char* outputFile = pos[1];
        char ppm_type;
        if (channels == 1) {
            ppm_type = '5';
        } else if (channels == 3) {
            ppm_type = '6';
        } else {
            throw std::invalid_argument("Unsupported channel count in .llc file (expected 1 or 3)");
        }
        Netpbm ppm(outputFile, ppm_type, width, height, 255);
        ppm.write(pixels.data(), pixels.size());
        ppm.close();
    } catch (const std::exception& e) {
        std::cerr << "Error decompressing image: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }

    return 0;
}
