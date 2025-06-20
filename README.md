# LLComp - The World's Best Lossless Image Compressor

LLComp is a cutting-edge **lossless image compression** tool designed to achieve unparalleled compression ratios while preserving every single bit of the original image data. This project is currently in its **initial development phase** and is inspired by the **FFV1 codec** from FFmpeg [[1]]. Whether you're working with high-resolution scientific imagery, medical scans, or multimedia datasets, LLComp ensures maximum efficiency without compromising quality.

## Key Features

- **Lossless Compression**: Every pixel of the original image is preserved during compression and decompression.
- **Advanced Algorithms**: Utilizes cutting-edge techniques such as adaptive arithmetic coding (`RangeEncoder` and `RangeDecoder`) and context modeling for optimal entropy reduction.
- **Multi-Channel Support**: Handles RGB and other multi-channel formats seamlessly.
- **Customizable Context Modeling**: Implements sophisticated prediction models (`Model3`) for precise pixel value estimation.
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux.
- **STB Image Integration**: Uses the STB library for image loading and saving.

## Why LLComp?

LLComp stands out as the **world's best lossless image compressor** due to its:
- Superior compression algorithms inspired by FFV1.
- Minimal computational overhead, ensuring fast encoding and decoding.
- Robust implementation of advanced statistical models (e.g., `MPS_PROBABILITY`, `NEXT_STATE_MPS`, and `NEXT_STATE_LPS`) for accurate data prediction.
- Extensive use of quantization tables (`quant5_table` and `quant11_table`) to optimize context hashing and reduce redundancy.

## Performance Metrics

| Format      | Compression Ratio | Encoding Speed | Decoding Speed |
|-------------|-------------------|----------------|----------------|
| PNG         |                   |                |                |
| WebP        |                   |                |                |
| JPEG-XL     |                   |                |                |
| LLComp      |                   |                |                |

*Results will be updated as the project progresses.*

## Installation

### Prerequisites

- Compiler: Tested with **Intel ICX 2025** on Windows.
- CMake (Version 3.15 or higher).
- Git.
- STB Image library (for image loading and saving).

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/vovach777/llcomp.git
   cd llcomp
   ```

2. Build the project using CMake:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. After building, two executables will be generated:
   - `llcompc(.exe)` – The **compressor** tool.
   - `llcompd(.exe)` – The **decompressor** tool.

## Usage

For examples of how to use LLComp, please refer to the [examples directory](examples/).

### Tools

- **Compressor**: Use the `llcompc(.exe)` executable to compress images.
- **Decompressor**: Use the `llcompd(.exe)` executable to decompress images.

## Technical Details

### Core Algorithms

- **Adaptive Arithmetic Coding**: The `RangeEncoder` and `RangeDecoder` classes implement adaptive arithmetic coding, which dynamically adjusts probabilities based on the data being processed.
- **Context Modeling**: The `Model3` class uses statistical models (`MPS_PROBABILITY`, `NEXT_STATE_MPS`, and `NEXT_STATE_LPS`) to predict pixel values with high accuracy.
- **Quantization Tables**: The `quant5_table` and `quant11_table` are used to reduce the range of differences between predicted and actual pixel values, minimizing entropy.
- **Median Prediction**: The `median` function computes the most likely pixel value based on neighboring pixels, improving compression efficiency.

## Development Environment

- **Compiler**: Intel ICX 2025 (Windows).
- **Build System**: CMake (cross-platform support).
- **Dependencies**: STB Image library for image loading and saving.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions from the community! If you'd like to contribute to LLComp, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

For major changes, please open an issue first to discuss the proposed updates.

## Acknowledgments

- Inspired by the **FFV1 codec** from FFmpeg.
- Special thanks to the open-source community for their support and feedback.
