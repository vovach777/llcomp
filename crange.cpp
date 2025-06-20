#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include "range.hpp"
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
    auto stb_img = stbi_load(argv[1] , &width, &height, &channels, 0);
    if (stb_img == nullptr) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::vector<uint8_t> rgb = std::vector<uint8_t>(stb_img, stb_img + width * height * channels);
    stbi_image_free(stb_img);

    std::vector<uint8_t> compressed = compressImage(rgb, width, height, channels);
    std::string outputFile = std::string(filename) + ".ppm";
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening output file: " << outputFile << std::endl;
        return 1;
    }
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    outFile.close();
    return 0;
}
