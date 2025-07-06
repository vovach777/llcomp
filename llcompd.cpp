#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include "llcomp.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image> [output_image]" << std::endl;
        std::cerr << "Output format can be 'png' or 'bmp' {default: png}" << std::endl;
        return 1;
    }
    std::string inFilename = std::string(argv[1]);
    std::string outFilename;
    std::ifstream inFile(inFilename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening input file: " << inFilename << std::endl;
        return 1;
    }
    std::vector<uint8_t> compressed((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();

    try {
        auto [pixels, width, height, channels] = decompressImage(compressed);

        std::string outFile;
        if (argc == 3)
            outFile = std::string(argv[2]);
        else
            outFile = std::string(inFilename) + ".png";

        if (outFile.substr(outFile.length()-4, 4) == ".bmp") {
            if (!stbi_write_bmp(outFile.c_str(), width, height, channels, pixels.data()))
                std::cerr << "Error writing output file: " << outFile << std::endl;
            }
        else {
            if (!stbi_write_png(outFile.c_str(), width, height, channels, pixels.data(), width * channels))
                std::cerr << "Error writing output file: " << outFile << std::endl;
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
