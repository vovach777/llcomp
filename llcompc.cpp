#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include "llcomp.hpp"
#include "profiling.hpp"

int main(int argc, char** argv) {
    profiling::StopWatch execute_time;
    execute_time.startnew();
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    try {
    const char* filename = argv[1];
    std::vector<uint8_t> compressed;
    llcomp::RawImage img;
    img.load(std::string(filename));
    img.le();
    switch (img.bits_per_channel)
    {
        case 8:  {
            compressed = llcomp::compressImage<8,8>(img.as<uint8_t>(), img.width, img.height, img.channel_nb );
            break;
        }
        case 10:  {
            compressed = llcomp::compressImage<10,10>(img.as<uint16_t>(), img.width, img.height, img.channel_nb );
            break;
        }
        case 12: {
            compressed = llcomp::compressImage<12,12>(img.as<uint16_t>(), img.width, img.height, img.channel_nb );
            break;
        }
        case 14: {
            compressed = llcomp::compressImage<14,14>(img.as<uint16_t>(), img.width, img.height, img.channel_nb );
            break;
        }
        case 15: {
            compressed = llcomp::compressImage<15,15>(img.as<uint16_t>(), img.width, img.height, img.channel_nb );
            break;
        }
        case 16: {
            compressed = llcomp::compressImage<16,16>(img.as<uint16_t>(), img.width, img.height, img.channel_nb );
            break;
        }
        default: {
            throw std::runtime_error("fail to load image: unknown bits per channal value!");
        }
    }

    std::string outputFile = std::string(filename) + llcomp::ext;
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening output file: " << outputFile << std::endl;
        return 1;
    }
    outFile.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    outFile.close();
    } catch (const std::exception& e) {
        std::cerr << "Error decompressing image: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }
    execute_time.stop();
    std::cout << "time: " << execute_time.elapsed_str() << std::endl;
    return 0;
}
