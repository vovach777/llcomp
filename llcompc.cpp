#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include "llcomp.hpp"
#define STB_IMAGE_IMPLEMENTATION
   #define STBI_NO_GIF
   #define STBI_NO_PSD
   #define STBI_NO_PIC
#include "stb_image.h"


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    const char* filename = argv[1];
    int width, height, channels;
    std::vector<uint8_t> compressed;
    if (stbi_is_16_bit(filename))
    {
        const uint16_t* stb_img = stbi_load_16(filename , &width, &height, &channels, 0);
        if (stb_img == nullptr) {
            std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
            return 1;
        }
        compressed = llcomp::compressImage(stb_img, uint32_t(width), uint32_t(height), uint32_t(channels),15 );
        stbi_image_free((stbi_us *)stb_img);
    } else {
        auto stb_img = stbi_load(filename , &width, &height, &channels, 0);
        if (stb_img == nullptr) {
            std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
            return 1;
        }
        compressed = llcomp::compressImage(stb_img, uint32_t(width), uint32_t(height), uint32_t(channels),8 );
        stbi_image_free((stbi_us *)stb_img);
    }

    std::string outputFile = std::string(filename) + llcomp::ext;
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening output file: " << outputFile << std::endl;
        return 1;
    }
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    outFile.close();
    return 0;
}
