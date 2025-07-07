#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include "llcomp.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


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
        auto [pixels, width, height, channels] = decompressImage(compressed);

        std::string outputFile = std::string(filename) + ".png";
        if  (!stbi_write_png(outputFile.c_str(), width, height, channels, pixels.data(), width * channels)) {
            std::cerr << "Error writing output file: " << outputFile << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error decompressing image: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }

    return 0;
}
