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


int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image> [output_umage]" << std::endl;
        return 1;
    }
    std::string inFilename = std::string(argv[1]);
    std::string outFilename;
    int width, height, channels;

    auto stb_img = stbi_load(argv[1] , &width, &height, &channels, 0);
    if (stb_img == nullptr) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::vector<uint8_t> rgb = std::vector<uint8_t>(stb_img, stb_img + width * height * channels);
    stbi_image_free(stb_img);

    std::vector<uint8_t> compressed = compressImage(rgb, width, height, channels);
    if (argc == 3)
        outFilename = std::string(argv[2]);
    else
        outFilename = std::string(inFilename) + ".llc";
    std::ofstream outFile(outFilename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening output file: " << outFilename << std::endl;
        return 1;
    }
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    outFile.close();
    return 0;
}
