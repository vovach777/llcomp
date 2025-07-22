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
    // if (llcomp::binarization::ilog2_32<0>(uint32_t{1}) == 0) {
    //     std::cerr << "Unsafe behavior is enabled" << std::endl;
    //     return;
    // }
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    const char* filename = argv[1];
    int width, height, channels;
    auto stb_img = stbi_load(argv[1] , &width, &height, &channels, 0);
    if (stb_img == nullptr) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::vector<uint8_t> rgb = std::vector<uint8_t>(stb_img, stb_img + width * height * channels);
    stbi_image_free(stb_img);

    std::vector<uint8_t> compressed = llcomp::compressImage(rgb.data(), uint32_t(width), uint32_t(height), uint32_t(channels) );
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
