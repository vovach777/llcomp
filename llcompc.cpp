#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include "llcomp.hpp"
#include "netpbm.hpp"


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_file>" << std::endl;
        return 1;
    }
    Netpbm ppm(argv[1]);
    uint32_t width = ppm.width;
    uint32_t height = ppm.height;
    uint32_t channels = (ppm.type == '6') ? 3 : (ppm.type == '5') ? 1 : 3;
    std::vector<uint8_t> rgb(width * height * channels);
    ppm.read(rgb.data(), rgb.size());
    ppm.close();

    std::vector<uint8_t> compressed = llcomp::compressImage(rgb, width, height, channels);
    const char* outputFile = argv[2];
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening output file: " << outputFile << std::endl;
        return 1;
    }
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    outFile.close();
    return 0;
}
