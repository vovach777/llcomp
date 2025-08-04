#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include "llcomp.hpp"
#include "profiling.hpp"

int main(int argc, char** argv) {
    profiling::StopWatch execute_time;
    execute_time.startnew();
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
    size_t size = inFile.seekg(0, std::ios::end).tellg() / sizeof(uint64_t);
    inFile.seekg(0, std::ios::beg);
    std::vector<uint64_t> compressed(size);
    inFile.read(reinterpret_cast<char*>(compressed.data()), size * sizeof(uint64_t));
    inFile.close();
    try {
        auto rawImage = llcomp::decompressImage(compressed.data(), compressed.data()+compressed.size() );
        execute_time.stop();
        rawImage.save(std::string(filename) + ".ppm");

    } catch (const std::exception& e) {
        std::cerr << "Error decompressing image: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }

    std::cout << "time: " << execute_time.elapsed_str() << std::endl;
    return 0;
}
