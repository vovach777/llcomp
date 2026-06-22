# LLComp — Lossless Image Compression

LLComp is a **lossless** image compression tool inspired by the **FFV1 codec** from FFmpeg [[1]](#1). The project started as a JavaScript prototype ([`llcomp.js`](llcomp.js)) and evolved into the current C++ implementation.

## Features

- **Lossless** — every pixel of the original image is preserved during compression and decompression
- **Adaptive arithmetic coding** — `RangeEncoder` / `RangeDecoder` with a CABAC state machine
- **Context modeling** — pixel prediction based on neighbor gradients with quantization tables (`quant5_table`, `quant7_table`, `quant11_table`)
- **Reversible Color Transform (RCT)** — for RGB images, the R and B channels are replaced by their differences from G (`r -= g; b -= g`), and G is adjusted from those differences; similar to the JPEG-LS transform
- **Three model sizes** — a choice between compression ratio and memory usage (speed is practically identical across models, see benchmarks):
  - **Small** — 7³ contexts, 2 lines (less memory)
  - **Standard** — 11³ contexts, 2 lines
  - **Large** — 11³×5×5 contexts, 3 lines (best compression, more memory)
- **Multi-channel support** — RGB, RGBA, grayscale (PGM/PPM)
- **Cross-platform** — Windows, macOS, Linux

## Project Structure

```
llcomp/
├── CMakeLists.txt      # C++ build (CMake 3.21+, C++17)
├── llcomp.hpp          # Header: codec core (C++)
├── llcompc.cpp         # Compressor CLI (C++)
├── llcompd.cpp         # Decompressor CLI (C++)
├── netpbm.hpp          # Netpbm I/O (PPM P6, PGM P5)
├── llcomp.js           # JavaScript prototype (Node.js + sharp)
├── README.md
└── LICENSE
```

## Implementations

### C++ (current)

The CLI tools work with the **Netpbm** format (PPM P6 for RGB, PGM P5 for grayscale).
Image preparation: `ffmpeg -i input.png -pix_fmt rgb24 output.ppm`

Generated executables:

| Executable        | Model       | Purpose          |
|-------------------|-------------|------------------|
| `llcompc`         | Standard    | Compression      |
| `llcompd`         | Standard    | Decompression    |
| `llcompc-s`       | Small       | Compression      |
| `llcompd-s`       | Small       | Decompression    |
| `llcompc-xl`      | Large       | Compression      |
| `llcompd-xl`      | Large       | Decompression    |

### JavaScript (prototype)

The file [`llcomp.js`](llcomp.js) is the **prototype** the project started from — an early development remnant. It is a full codec implementation in JavaScript using the **sharp** library for image I/O (PNG, JPEG, etc.), with asynchronous execution that yields to the event loop. The C++ version is the primary, maintained implementation.

```bash
# Compress
node llcomp.js image.png

# Decompress
node llcomp.js image.png.llcomp
```

## `.llc` File Format

Compressed files use the `.llc` extension and contain:

| Offset | Size | Description                          |
|--------|------|-------------------------------------|
| 0      | 1    | Magic number + revision (0x7B)      |
| 1      | 1    | Number of channels (1, 3, 4)        |
| 2      | 2    | Width (little-endian, max 65535)    |
| 4      | 2    | Height (little-endian, max 65535)   |
| 6      | ...  | Compressed data (arithmetic coded)  |

> **Note:** The format depends on the model size — a file compressed with one model cannot be decompressed with another.

## Installation & Build (C++)

### Requirements

- C++17 compiler (tested with **Intel ICX 2025** on Windows)
- CMake 3.21 or higher
- Git

### Build

```bash
git clone https://github.com/vovach777/llcomp.git
cd llcomp
mkdir build && cd build
cmake ..
cmake --build .
```

## Usage (C++)

```bash
# Compress (Standard)
llcompc input.ppm output.llc

# Decompress (Standard)
llcompd output.llc restored.ppm

# Compress (Large — better compression, more memory)
llcompc-xl input.ppm output.llc

# Compress (Small — less memory, worse compression)
llcompc-s input.ppm output.llc
```

## Benchmarks

Test set: 10 images, ~70 MB total original size.

### Compression Ratio

| File                                | Original        | LLC-Small       | BPP    | LLC             | BPP    | LLC-Large       | BPP    | HALIC           | BPP    |
|-------------------------------------|-----------------|-----------------|--------|-----------------|--------|-----------------|--------|-----------------|--------|
| pawel-kadysz-3295.pnm               | 9,437,201       | 3,696,785       | 9.40   | 3,664,218       | 9.32   | 3,634,226       | 9.24   | 3,784,562       | 9.62   |
| jeff-sheldon-3224.pnm               | 9,000,017       | 2,804,760       | 7.48   | 2,768,315       | 7.38   | 2,748,357       | 7.33   | 2,872,223       | 7.66   |
| mike-petrucci-25087.pnm             | 5,992,721       | 2,043,043       | 8.18   | 2,017,585       | 8.08   | 2,010,085       | 8.05   | 2,289,433       | 9.17   |
| andre-robillard-298.pnm             | 8,116,241       | 2,431,494       | 7.19   | 2,393,436       | 7.08   | 2,371,957       | 7.01   | 2,573,740       | 7.61   |
| jimmy-musto-39162.pnm               | 8,386,577       | 3,808,255       | 10.90  | 3,732,003       | 10.68  | 3,655,473       | 10.46  | 4,308,221       | 12.33  |
| austin-neill-189141.pnm             | 8,382,482       | 3,153,655       | 9.03   | 3,124,780       | 8.95   | 3,106,723       | 8.89   | 3,312,174       | 9.48   |
| ryan-pohanic-13342.pnm              | 8,386,577       | 3,856,674       | 11.04  | 3,836,933       | 10.98  | 3,822,185       | 10.94  | 3,845,120       | 11.00  |
| leeroy-653.pnm                      | 1,843,215       | 1,092,627       | 14.23  | 1,091,467       | 14.21  | 1,100,860       | 14.33  | 1,092,349       | 14.22  |
| steve-richey-141.pnm                | 6,436,817       | 2,928,559       | 10.92  | 2,904,952       | 10.83  | 2,894,947       | 10.79  | 3,129,741       | 11.67  |
| artur-pokusin-739.pnm               | 7,558,289       | 2,266,552       | 7.20   | 2,247,960       | 7.14   | 2,242,602       | 7.12   | 2,199,231       | 6.98   |
| **Total**                           | **73,540,137**  | **28,082,404**  | **9.16** | **27,781,649** | **9.07** | **27,587,415** | **9.00** | **29,406,794** | **9.60** |

### Encoding Speed (10 files, 70.13 MB)

| Codec           | Time (s)   | Speed (MB/s) |
|-----------------|------------|--------------|
| LLC-Small       | 5.14       | 13.65        |
| LLC             | 4.92       | 14.27        |
| LLC-Large       | 5.25       | 13.36        |
| HALIC           | 0.48       | 146.30       |

### Decoding Speed

| Codec           | Time (s)   | Speed (MB/s) |
|-----------------|------------|--------------|
| LLC-Small       | 5.12       | 13.70        |
| LLC             | 5.05       | 13.90        |
| LLC-Large       | 5.36       | 13.08        |
| HALIC           | 0.58       | 119.92       |

## Technical Details

### Compression Algorithm

1. **Reversible Color Transform (RCT)** — for RGB, a reversible transform is applied: R and B are replaced by their differences from G (`r -= g; b -= g`), then G is adjusted: `g += (b + r) >> 2`. Similar to JPEG-LS
2. **Prediction** — median predictor: `median(L, L + T - TL, T)` based on neighboring pixels
3. **Context** — a hash of quantized gradients of neighboring pixels; the size depends on the model
4. **Binarization** — exponential-Golomb-like residual coding (`putSymbol` / `getSymbol`)
5. **Arithmetic coding** — bit-wise coding with adaptive probabilities via a CABAC state machine

### Key Components (C++)

- [`RangeEncoder`](llcomp.hpp:51) / [`RangeDecoder`](llcomp.hpp:109) — arithmetic coding
- [`cabac::State`](llcomp.hpp:312) — state machine with transition tables `nextStateMps` / `nextStateLps`
- [`binarization::putSymbol`](llcomp.hpp:195) / [`binarization::getSymbol`](llcomp.hpp:248) — symbol binarization/decoding
- [`quant11()`](llcomp.hpp:364) / [`quant5()`](llcomp.hpp:368) / [`quant7()`](llcomp.hpp:372) — gradient quantization functions
- [`median()`](llcomp.hpp:381) — median predictor
- [`compressImage()`](llcomp.hpp:386) / [`decompressImage()`](llcomp.hpp:510) — main compression/decompression functions

## Development Environment

- **Compiler**: Intel ICX 2025 (Windows)
- **Build**: CMake 3.21+, C++17
- **C++ dependencies**: none third-party (standard library only)
- **JS dependencies**: Node.js, [sharp](https://www.npmjs.com/package/sharp)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References

<a id="1">[1]</a> FFV1 codec — https://ffmpeg.org/ffmpeg-codecs.html#libavcodec_002dhuffyuv
