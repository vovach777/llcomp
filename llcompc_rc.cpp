#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include <thread>
#include <algorithm>
#include "llcomp_rc.hpp"
#include "netpbm.hpp"


int main(int argc, char** argv) {
    // [-s N|c] equal strips: N fixed (block 64 px granularity, widths differ by <= 64), c = number of HW threads,
    //          default 0 = auto W/512;  [-t T] worker threads (default: HW threads)
    int nstrips = 0, threads = int(std::thread::hardware_concurrency()); std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-s" && i + 1 < argc) { std::string v = argv[++i]; nstrips = v == "c" ? int(std::thread::hardware_concurrency()) : std::stoi(v); }
        else if (a == "-t" && i + 1 < argc) threads = std::max(1, std::stoi(argv[++i]));
        else pos.push_back(argv[i]);
    }
    if (pos.size() < 2) {
        std::cerr << "Usage: " << argv[0] << " [-s N|c] [-t threads] <input_image> <output_file>\n"
                     "  -s N  split into N equal strips (2, 4, ...; widths in 64-px blocks), c = HW threads, 0 = auto W/512" << std::endl;
        return 1;
    }
    Netpbm ppm(pos[0]);
    uint32_t width = ppm.width;
    uint32_t height = ppm.height;
    uint32_t channels = (ppm.type == '6') ? 3 : (ppm.type == '5') ? 1 : 3;
    std::vector<uint8_t> rgb(width * height * channels);
    ppm.read(rgb.data(), rgb.size());
    ppm.close();

    std::vector<uint8_t> compressed = llcomp_rc::compressImage(rgb, width, height, channels, threads, nstrips);
    const char* outputFile = pos[1];
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening output file: " << outputFile << std::endl;
        return 1;
    }
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    outFile.close();
    return 0;
}
