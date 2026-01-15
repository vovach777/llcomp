# LLR - Simple solution good compression

LLRice is streming frendly intraframe lossless image compressor.

## Key Features

- **Lossless Compression**: Every pixel of the original image is preserved during compression and decompression.
- **Advanced Algorithms**: RGB -> RCT -> LOCO-I predictor -> error -> RLGR -> Interleaved Bitstream
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux.

## Why LLRice?

 Simple solution good compression and speed

## Performance Metrics

| File                                | Original        | LLRice          | BPP    | FFV1            | BPP    | HALIC           | BPP    | Status     |
|-------------------------------------|-----------------|-----------------|--------|-----------------|--------|-----------------|--------|------------|
| abigail-keenan-27293.pnm            | 8,386,577       | 3,290,640       | 9.42   | 3,145,070       | 9.00   | 3,269,482       | 9.36   | OK         |
| adam-przewoski-193.pnm              | 4,148,945       | 417,984         | 2.42   | 380,715         | 2.20   | 480,670         | 2.78   | OK         |
| adam-willoughby-knox-56406.pnm      | 8,355,857       | 2,726,000       | 7.83   | 2,573,739       | 7.39   | 2,673,948       | 7.68   | OK         |
| afroz-nawaf-323.pnm                 | 8,386,577       | 3,623,880       | 10.37  | 3,457,772       | 9.90   | 3,611,203       | 10.33  | OK         |
| alberto-restifo-4549.pnm            | 8,392,721       | 2,918,592       | 8.35   | 2,744,693       | 7.85   | 2,838,822       | 8.12   | OK         |
| alec-weir-22151.pnm                 | 6,319,121       | 1,431,248       | 5.44   | 1,393,516       | 5.29   | 1,489,874       | 5.66   | OK         |
| alejandro-escamilla-1.pnm           | 8,386,577       | 1,909,096       | 5.46   | 1,764,633       | 5.05   | 1,970,349       | 5.64   | OK         |
| alejandro-escamilla-10.pnm          | 8,386,577       | 1,810,592       | 5.18   | 1,597,089       | 4.57   | 1,842,802       | 5.27   | OK         |
| alejandro-escamilla-2.pnm           | 8,429,585       | 2,082,072       | 5.93   | 1,909,184       | 5.44   | 2,093,252       | 5.96   | OK         |
| alejandro-escamilla-23.pnm          | 10,579,985      | 4,683,504       | 10.62  | 4,484,474       | 10.17  | 4,411,476       | 10.01  | OK         |
| alejandro-escamilla-25.pnm          | 4,530,065       | 2,153,736       | 11.41  | 2,045,973       | 10.84  | 2,159,913       | 11.44  | OK         |
| aleks-dorohovich-26.pnm             | 6,776,672       | 2,117,992       | 7.50   | 1,946,562       | 6.89   | 2,151,463       | 7.62   | OK         |
| aleks-dorohovich-38.pnm             | 7,527,185       | 3,666,752       | 11.69  | 3,476,683       | 11.09  | 3,669,730       | 11.70  | OK         |
| aleksandra-boguslawska-288.pnm      | 8,355,857       | 4,433,816       | 12.73  | 4,319,264       | 12.41  | 4,364,448       | 12.54  | OK         |
| aleksi-tappura-445.pnm              | 8,331,281       | 2,900,848       | 8.36   | 2,677,278       | 7.71   | 2,825,478       | 8.14   | OK         |
| ales-krivec-1211.pnm                | 8,331,281       | 3,129,992       | 9.02   | 2,924,753       | 8.43   | 3,191,846       | 9.19   | OK         |
| ales-krivec-2051.pnm                | 8,331,281       | 4,097,480       | 11.80  | 3,967,257       | 11.43  | 4,046,975       | 11.66  | OK         |
| ales-krivec-26511.pnm               | 8,828,945       | 3,537,480       | 9.62   | 3,365,196       | 9.15   | 3,513,183       | 9.55   | OK         |
| ales-krivec-2892.pnm                | 8,398,865       | 5,167,216       | 14.77  | 5,021,537       | 14.35  | 5,049,569       | 14.43  | OK         |
| ales-krivec-3727.pnm                | 8,398,865       | 4,469,800       | 12.77  | 4,302,599       | 12.29  | 4,432,621       | 12.67  | OK         |
| ales-krivec-3728.pnm                | 8,398,865       | 3,480,496       | 9.95   | 3,279,143       | 9.37   | 3,418,525       | 9.77   | OK         |
| ales-krivec-434.pnm                 | 8,331,281       | 3,242,384       | 9.34   | 3,060,549       | 8.82   | 3,275,702       | 9.44   | OK         |
| ales-krivec-731.pnm                 | 8,331,281       | 3,068,544       | 8.84   | 2,922,459       | 8.42   | 3,020,987       | 8.70   | OK         |
| alex-siale-95113.pnm                | 8,386,577       | 4,123,232       | 11.80  | 4,007,795       | 11.47  | 4,144,258       | 11.86  | OK         |
|-------------------------------------|-----------------|-----------------|--------|-----------------|--------|-----------------|--------|------------|
| **TOTAL**                           | **191,030,823** | **74,483,376**  | **9.36** | **70,767,933** | **8.89** | **73,946,576** | **9.29** |            |

*Results as of January 2026, comparing LLR with FFV1 and HALIC.*

## Speed (585 files):

| Operation            | Time (s)     | Speed (MB/s)    |
|----------------------|--------------|-----------------|
| LLR Compression      | 110.5588     | 35.30           |
| LLR Decompression    | 101.0031     | 38.64           |
|----------------------|--------------|-----------------|

## Installation

### Prerequisites

- Compiler: Tested with **Intel ICX 2025** on Windows. mingw. clang.
- CMake (Version 3.15 or higher).
- Git.

### Steps


1. After building, two executables will be generated:
   - `llrice-c(.exe)` – The **compressor** tool.
   - `llrice-d(.exe)` – The **decompressor** tool.

## Usage

For examples of how to use LLRice, please refer to the [examples directory](examples/).

### Tools

- **Compressor**: Use the `llrice-c(.exe)` executable to compress images.
- **Decompressor**: Use the `llrice-d(.exe)` executable to decompress images.

## Technical Details

### Core Algorithms

- **RLGR**: by microsoft....
- **Iterleaved Deterministic BitStream**: auto maneged BitStreams.
- **Prediction**: same FFV1

## Development Environment

- **Compiler**: Intel ICX 2025 (Windows).
- **Build System**: CMake (cross-platform support).
- **Dependencies**: STB Image library for image loading and saving.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions from the community! If you'd like to contribute to LLRice, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

For major changes, please open an issue first to discuss the proposed updates.

## Acknowledgments

- Inspired by the **FFV1 codec** from FFmpeg.
- Special thanks to the open-source community for their support and feedback.
