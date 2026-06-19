#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include "llcomp.hpp"
#include "netpbm.hpp"


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_image>" << std::endl;
        return 1;
    }
    std::ifstream inFile(argv[1], std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening input file: " << argv[1] << std::endl;
        return 1;
    }
    std::vector<uint8_t> compressed((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();

    try {
        auto [pixels, width, height, channels] = llcomp::decompressImage(compressed);

        const char* outputFile = argv[2];
        char ppm_type = (channels == 3) ? '6' : (channels == 1) ? '5' : '6';
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
