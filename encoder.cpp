#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include "llrice.hpp"
#include "profiling.hpp"
#include "image.hpp"

int main(int argc, char** argv) {
    profiling::StopWatch execute_time;

    if (argc < 3) {
        std::cout << "LLR - Simple solution good compression" << std::endl;
        std::cout << "LLRice is streming frendly intraframe lossless image compressor." << std::endl;
        std::cout << "Only .ppm (P6) types are supported for input!" << std::endl;
        std::cout << "Please use this format: \"" << "llrice-c" << " input_file output_file\"" << std::endl;
        return 0;
    }
    try {
        const char* filename = argv[1];
        const char* outputFilename = argv[2];
        std::vector<uint64_t> compressed;
        RawImage img;
        img.load(std::string(filename));
        img.le();
        execute_time.startnew();
        compressed = llrice::compressImage(img );

        std::ofstream outFile(outputFilename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error opening output file: " << outputFilename << std::endl;
            return 1;
        }
        outFile.write(reinterpret_cast<const char*>(compressed.data()), compressed.size()*sizeof(uint64_t));
        outFile.close();
    } catch (const std::exception& e) {
        std::cerr << "Error compressing image: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }
    execute_time.stop();
    //std::cout << "time: " << execute_time.elapsed_str() << std::endl;
    return 0;
}
