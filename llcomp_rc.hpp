// llcomp RC engine: llcomp with the binary CABAC replaced by the PLT2 back-end, on the tiny context.
//
//  per pixel, per channel:  RCT (RGB: Y = G + ((R-G + B-G) >> 2), U = B-G, V = R-G; gray / alpha as is)
//   -> MED prediction (PALIC border rule)
//   -> tiny context: q5 of the gradients g1 = a-c, g2 = c-b, g3 = d-b (levels 0, +-1, +-2; thresholds luma (0,3),
//      chroma (0,4)), mirrored by sign -> 63 cells, x Enb = min(8, bit_width(max|e_left|, |e_top|)) -> 567 contexts
//   -> PALIC-2 alphabet A31: letter (0, +-15 magnitude classes) + raw low bits of the class (bypass)
//   -> letter: adaptive CDF (32 x u16, total 2^15, AV1-style update, uniform start, rate 1/16 .. 1/256)
//      + range coder (low 64 / range 32, byte renorm, HYBRID carry: 4 pending bytes in low + Subbotin trim
//      when all are 0xFF -> every emitted byte is final);  bypass: one plain bit stream (64-bit words).
//  Models: Y own, U and V share one, alpha own; gray = one luma channel.
//
//  Strips: N = max(1, W/512) (or set), floor(W/64) blocks of 64 px dealt out evenly; strips are independent
//  (own coders, models, borders) -> threads. One buffer per strip: the channels' range streams + the shared bypass
//  stream interleaved by deterministic slot reservation (RC: 4 bytes at start, 1 per renorm shift; bit writer:
//  8 bytes when the first bit of a new word is written). Nothing is moved or glued except strip concatenation.
//
//  Context modes 0..3 (-x): see nctx_of below; 0 = tiny (default, old files).
//  File: magic (0x77 + 16) | channels + (mode << 4) | u16 W | u16 H (as llcomp) | u8 strips (0 = W/512) | u32 len[N] | strips.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__AVX512BW__) && !defined(LLC_SCALAR)
#include <immintrin.h>
#define LLC_SIMD 2
#elif defined(__AVX2__) && !defined(LLC_SCALAR)
#include <immintrin.h>
#define LLC_SIMD 1
#elif defined(__SSE4_1__) && !defined(LLC_SCALAR) && !defined(LLC_FORCE_SIMDE)   // Nehalem..Ivy Bridge: native 4 x XMM
#include <immintrin.h>
#define LLC_SIMD 3
#elif !defined(LLC_SCALAR)
#if !defined(__AVX__)                   // __m256i, _mm256_load/store/set* are AVX1: needed below AVX (core2),
#define SIMDE_X86_AVX_ENABLE_NATIVE_ALIASES // but clash with native ones on AVX1 (sandybridge)
#endif
#define SIMDE_X86_AVX2_ENABLE_NATIVE_ALIASES
#include "avx2.h"
#define LLC_SIMD 1
#else
#define LLC_SIMD 0
#endif
#include "llcomp_rc_tables.h"
#include "llcomp_fse_groups.h"   // kGroups: PLT2 4-D cell -> 8 groups (CLIC calibration)
#include "llcomp_large_map.h"    // kLargeMap: FFV1-large-like cell -> 256 clusters (llcomp_large_cluster.py)
#include <cstdio>
#include <cstdlib>

namespace llcomp_rc {
using namespace llcomp_rc_tab;
#define LLC_AI __attribute__((always_inline)) inline
constexpr uint8_t magic_rc = 0x77 + 16;

// ------------------------------------------------------------------ static tables
struct Static {
    uint16_t bw8[512];
    alignas(64) uint16_t tle[32], tgt[32];
    Static() {
        for (int v = 0; v < 512; ++v) { int b = v ? 32 - __builtin_clz(unsigned(v)) : 0; bw8[v] = uint16_t(b > 8 ? 8 : b); }
        for (int i = 0; i < 32; ++i) { tle[i] = uint16_t((i < 31 ? i : 31) * 2); tgt[i] = uint16_t(32768 - (31 - i > 0 ? 31 - i : 0) * 2); }
    }
};
inline const Static& S() { static const Static s; return s; }

// ------------------------------------------------------------------ adaptive CDF (31 letters, cum[31] = 32768)
constexpr int NL = 31, R0 = 4, RMAX = 8;
// context modes (-x, stored in the high nibble of the channels byte; 0 = the original tiny context, old files):
//  0 tiny: q5 mirrored (63) x Enb (9)                                    567
//  1 4d:   PLT2 q7(c-a) x q7(c-b) x q7(d-b), mirrored by the tiny sign, x error level of
//          max(z top-left, z top, z top-right) (zigzag of s*e; thresholds Y 8 32, U/V 3 8)         343 x 3
//  2 4dE:  the same q7 gradients x llcomp's Enb (9)                                                343 x 9
//  3 g8:   mode 1 unmirrored -> the 8 FSE groups of PLT2 (what the static coder uses)              8
//  4 L256: FFV1-large-like cell q11(L-TL), q11(TL-T), q11(T-TR), q5(LL-L), q5(TT-T) (16638 mirrored cells)
//          clustered to 256 contexts by a static map trained on a corpus (llcomp_large_cluster.py)       256
//  5 L256E: the same 256 clusters x Enb (9)                                                          2304
//  6 L256P: mode 4 with preset CDFs: every cluster starts from its corpus letter distribution (kLargePreset)
//          and a slower start of the adaptation (count kPresetCnt)                                      256
constexpr int NMODES = 7;
constexpr int nctx_of(int mode) { return mode == 0 ? 63 * 9 : mode == 1 ? 343 * 3 : mode == 2 ? 343 * 9 : mode == 3 ? 8 : mode == 5 ? 256 * 9 : 256; }
constexpr int kPresetCnt = 64;                                   // preset start: rate 6 (tuned held-out: 16..128 within 0.1%)
inline int preset_cnt() { return kPresetCnt; }
constexpr int NLCELL = 16638;
inline FILE* dump_file() { static FILE* f = getenv("LLC_DUMP") ? fopen(getenv("LLC_DUMP"), "ab") : nullptr; return f; }   // (cell, letter, plane)
static inline int ql11(int g) { int a = g < 0 ? -g : g, v = (a >= 1) + (a >= 2) + (a >= 5) + (a >= 12) + (a >= 23); return g < 0 ? -v : v; }
static inline int ql5(int g) { int a = g < 0 ? -g : g, v = (a >= 1) + (a >= 4); return g < 0 ? -v : v; }
struct Q7T { int8_t q[2][2048]; Q7T() { static const int E[2][3] = {{1, 3, 10}, {1, 3, 7}};
    for (int t = 0; t < 2; ++t) for (int g = -1024; g < 1024; ++g) { int a = g < 0 ? -g : g, v = (a >= E[t][0]) + (a >= E[t][1]) + (a >= E[t][2]); q[t][g + 1024] = int8_t(g < 0 ? -v : v); } } };
inline const Q7T& Q7() { static const Q7T t; return t; }
struct alignas(64) Cdf { uint16_t c[32]; };
LLC_AI int rate_of(uint16_t& cnt) {
    int r = R0 + (cnt > 15) + (cnt > 31) + (cnt > 63) + (cnt > 127) + (cnt > 255) + (cnt > 511);
    cnt += cnt < 1024; return r < RMAX ? r : RMAX;
}
LLC_AI void cdf_update(Cdf& d, uint16_t& cnt, int l) {
    int rate = rate_of(cnt); const Static& st = S();
#if LLC_SIMD == 2
    __m512i c = _mm512_load_si512(d.c); __mmask32 gt = ~((2u << l) - 1); __m128i sh = _mm_cvtsi32_si128(rate);
    __m512i up = _mm512_srl_epi16(_mm512_sub_epi16(_mm512_load_si512(st.tgt), c), sh);
    __m512i dn = _mm512_srl_epi16(_mm512_sub_epi16(c, _mm512_load_si512(st.tle)), sh);
    c = _mm512_mask_add_epi16(c, gt, c, up); c = _mm512_mask_sub_epi16(c, ~gt, c, dn); _mm512_store_si512(d.c, c);
#elif LLC_SIMD == 1
    const __m256i idx = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); __m128i sh = _mm_cvtsi32_si128(rate);
    for (int h = 0; h < 32; h += 16) {
        __m256i c = _mm256_load_si256((const __m256i*)(d.c + h));
        __m256i gt = _mm256_cmpgt_epi16(_mm256_add_epi16(idx, _mm256_set1_epi16(short(h))), _mm256_set1_epi16(short(l)));
        __m256i up = _mm256_srl_epi16(_mm256_sub_epi16(_mm256_load_si256((const __m256i*)(st.tgt + h)), c), sh);
        __m256i dn = _mm256_srl_epi16(_mm256_sub_epi16(c, _mm256_load_si256((const __m256i*)(st.tle + h))), sh);
        _mm256_store_si256((__m256i*)(d.c + h), _mm256_blendv_epi8(_mm256_sub_epi16(c, dn), _mm256_add_epi16(c, up), gt));
    }
#elif LLC_SIMD == 3
    const __m128i idx = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7), ll = _mm_set1_epi16(short(l)); __m128i sh = _mm_cvtsi32_si128(rate);
    for (int h = 0; h < 32; h += 8) {
        __m128i c = _mm_load_si128((const __m128i*)(d.c + h));
        __m128i gt = _mm_cmpgt_epi16(_mm_add_epi16(idx, _mm_set1_epi16(short(h))), ll);
        __m128i up = _mm_srl_epi16(_mm_sub_epi16(_mm_load_si128((const __m128i*)(st.tgt + h)), c), sh);
        __m128i dn = _mm_srl_epi16(_mm_sub_epi16(c, _mm_load_si128((const __m128i*)(st.tle + h))), sh);
        _mm_store_si128((__m128i*)(d.c + h), _mm_blendv_epi8(_mm_sub_epi16(c, dn), _mm_add_epi16(c, up), gt));
    }
#else
    for (int i = 1; i <= l; ++i) d.c[i] = uint16_t(d.c[i] - (uint16_t(d.c[i] - st.tle[i]) >> rate));
    for (int i = l + 1; i < NL; ++i) d.c[i] = uint16_t(d.c[i] + (uint16_t(st.tgt[i] - d.c[i]) >> rate));
#endif
}
LLC_AI int cdf_find(const Cdf& d, uint32_t v) {
#if LLC_SIMD == 2
    return __builtin_popcount(_mm512_cmple_epu16_mask(_mm512_load_si512(d.c), _mm512_set1_epi16(short(v)))) - 1;
#elif LLC_SIMD == 1
    __m256i vv = _mm256_set1_epi16(short(v)); int n = 0;
    for (int h = 0; h < 32; h += 16) n += __builtin_popcount(unsigned(_mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_max_epu16(_mm256_load_si256((const __m256i*)(d.c + h)), vv), vv))));
    return n / 2 - 1;
#elif LLC_SIMD == 3
    const __m128i vv = _mm_set1_epi16(short(v)); const __m128i* p = (const __m128i*)d.c;   // c[i] <= v  <=>  max(c[i], v) == v
    __m128i m0 = _mm_cmpeq_epi16(_mm_max_epu16(_mm_load_si128(p + 0), vv), vv), m1 = _mm_cmpeq_epi16(_mm_max_epu16(_mm_load_si128(p + 1), vv), vv);
    __m128i m2 = _mm_cmpeq_epi16(_mm_max_epu16(_mm_load_si128(p + 2), vv), vv), m3 = _mm_cmpeq_epi16(_mm_max_epu16(_mm_load_si128(p + 3), vv), vv);
    return __builtin_popcount(unsigned(_mm_movemask_epi8(_mm_packs_epi16(m0, m1))) | unsigned(_mm_movemask_epi8(_mm_packs_epi16(m2, m3))) << 16) - 1;
#else
    const uint16_t* c = d.c; int l = 0;
    l += (c[l + 16] <= v) ? 16 : 0; l += (c[l + 8] <= v) ? 8 : 0; l += (c[l + 4] <= v) ? 4 : 0; l += (c[l + 2] <= v) ? 2 : 0; l += (c[l + 1] <= v) ? 1 : 0;
    return l;
#endif
}
struct Model {
    std::vector<Cdf> cdf; std::vector<uint16_t> cnt;
    explicit Model(int n) : cdf(n), cnt(n, 0) { for (auto& d : cdf) { for (int i = 0; i < 31; ++i) d.c[i] = uint16_t(i * 32768 / 31); d.c[31] = 32768; } }
    void preset(int g) { for (size_t k = 0; k < cdf.size() && k < 256; ++k) { memcpy(cdf[k].c, llcomp_large::kLargePreset[g][k], 64); cnt[k] = uint16_t(preset_cnt()); } }
};

// ------------------------------------------------------------------ strip buffer, range coder (hybrid carry), bits
constexpr uint32_t TOP = 1u << 24;
struct Buf {
    std::vector<uint8_t> b; uint32_t next = 0;
    LLC_AI uint32_t take(uint32_t n) { uint32_t s = next; next += n; if (__builtin_expect(next + 16 > b.size(), 0)) b.resize(b.size() * 2 + 64); return s; }
};
struct RcEnc {
    uint64_t low = 0; uint32_t range = 0xFFFFFFFFu; int p = 0; uint32_t fifo[16]; uint32_t head = 0, tail = 0;
    void init(Buf& B) { for (int i = 0; i < 4; ++i) fifo[tail++ & 15] = B.take(1); }
    LLC_AI void put(Buf& B, uint32_t c, uint32_t f) {
        uint32_t r = range >> 15; low += uint64_t(c) * r; range = f * r;
        while (range < TOP) {
            if (p == 4) B.b[fifo[head++ & 15]] = uint8_t(low >> 56); else ++p;
            fifo[tail++ & 15] = B.take(1); low <<= 8; range <<= 8;
            if (__builtin_expect((low >> 32) == 0xFFFFFFFFu, 0)) { uint64_t lim = (1ull << 32) - (low & 0xFFFFFFFFu); if (range > lim) range = uint32_t(lim); }
        }
    }
    void flush(Buf& B) { uint64_t l = low << (8 * (4 - p)); for (int i = 0; i < p + 4; ++i) { B.b[fifo[head++ & 15]] = uint8_t(l >> 56); l <<= 8; } }
};
struct RcDec {
    uint64_t low = 0; uint32_t range = 0xFFFFFFFFu, code = 0;
    void init(const uint8_t* b, uint32_t& nx) { for (int i = 0; i < 4; ++i) code = (code << 8) | b[nx++]; }
    LLC_AI void step(uint32_t c, uint32_t f, uint32_t r, const uint8_t* b, uint32_t& nx) {
        code -= c * r; low += uint64_t(c) * r; range = f * r;
        while (range < TOP) {
            code = (code << 8) | b[nx++]; low <<= 8; range <<= 8;
            if (__builtin_expect((low >> 32) == 0xFFFFFFFFu, 0)) { uint64_t lim = (1ull << 32) - (low & 0xFFFFFFFFu); if (range > lim) range = uint32_t(lim); }
        }
    }
};
struct BitW {
    uint64_t acc = 0; int fill = 0; uint32_t slot = 0; bool has = false;
    LLC_AI void put(Buf& B, uint32_t v, int n) {
        if (!has) { slot = B.take(8); has = true; }
        acc |= uint64_t(v) << fill; fill += n;
        if (fill >= 64) { std::memcpy(&B.b[slot], &acc, 8); fill -= 64; if (fill) { slot = B.take(8); acc = uint64_t(v) >> (n - fill); } else { has = false; acc = 0; } }
    }
    void flush(Buf& B) { if (has) std::memcpy(&B.b[slot], &acc, 8); }
};
struct BitR {
    uint64_t cur = 0; int avail = 0;
    LLC_AI uint32_t get(const uint8_t* b, uint32_t& nx, int n) {
        if (avail >= n) { uint32_t v = uint32_t(cur) & ((1u << n) - 1); cur >>= n; avail -= n; return v; }
        uint64_t w; std::memcpy(&w, b + nx, 8); nx += 8;
        uint32_t v = uint32_t(cur | (w << avail)) & ((1u << n) - 1); int k = n - avail; cur = w >> k; avail = 64 - k; return v;
    }
};

// ------------------------------------------------------------------ channel (KIND 0: 8-bit luma-like, 1: 9-bit chroma)
template <int KIND, int MODE = 0>
struct Chan {
    static constexpr int G = KIND, HALF = KIND ? 256 : 128, MASK = KIND ? 511 : 255, MID = KIND ? 0 : 128, T1 = 0, T2 = KIND ? 4 : 3;
    static constexpr int VT1 = KIND ? 3 : 8, VT2 = KIND ? 8 : 32;          // modes 1, 3: error levels on z (zigzag)
    Model* M; std::vector<int16_t> ring; std::vector<uint16_t> e0, e1, z0, z1; int w; mutable int cell = 0;
    int16_t* P; const int16_t* Q; const int16_t* Q2; uint16_t* Ec; const uint16_t* Ep; uint16_t* Zc; const uint16_t* Zp;
    Chan(Model* m, int w_) : M(m), ring(3 * size_t(w_)), e0(w_), e1(w_), z0(size_t(w_) + 2, 0), z1(size_t(w_) + 2, 0), w(w_) {}
    LLC_AI void row(int y) { P = ring.data() + (y % 3) * w; Q = y ? ring.data() + ((y - 1) % 3) * w : P; Q2 = y >= 2 ? ring.data() + ((y - 2) % 3) * w : Q; Ec = (y & 1) ? e1.data() : e0.data(); Ep = (y & 1) ? e0.data() : e1.data();
        Zc = ((y & 1) ? z1.data() : z0.data()) + 1; Zp = ((y & 1) ? z0.data() : z1.data()) + 1;   // halo 0 at -1 and w; row -1 = 0
        if (y == 0) std::fill(z1.begin(), z1.end(), uint16_t(0)); }
    static LLC_AI int q5(int g) { int a = g < 0 ? -g : g, s = (g > 0) - (g < 0); return s * ((a > T1) + (a > T2)); }
    LLC_AI void ctx(int x, int y, int& pred, int& cx, int& s) const {
        int a, b, c, d, el = 0, et = 0;
        if (y == 0) { a = x ? P[x - 1] : MID; b = c = d = a; if (x) el = Ec[x - 1]; }
        else { b = Q[x]; if (x == 0) { a = c = b; } else { a = P[x - 1]; c = Q[x - 1]; el = Ec[x - 1]; } d = (x + 1 < w) ? Q[x + 1] : b; et = Ep[x]; }
        int mn = a < b ? a : b, mx = a < b ? b : a, pr = a + b - c; pred = pr < mn ? mn : (pr > mx ? mx : pr);
        if (MODE >= 4) {                                                    // 4, 5, 6
            const int ll = x >= 2 ? P[x - 2] : a, tt = y >= 2 ? Q2[x] : b;
            int idx = ql11(a - c) + 11 * ql11(c - b) + 121 * ql11(b - d) + 1331 * ql5(ll - a) + 6655 * ql5(tt - b);
            s = 1; if (idx < 0) { idx = -idx; s = -1; }                          // FFV1 mirror
            cell = idx; const int k = llcomp_large::kLargeMap[G][idx];
            cx = MODE == 5 ? k * 9 + (el > et ? el : et) : k;
        } else if (MODE == 0) {
            int idx = (q5(a - c) + 2) * 25 + (q5(c - b) + 2) * 5 + (q5(d - b) + 2);
            s = 1; if (idx > 62) { idx = 124 - idx; s = -1; }
            cx = idx * 9 + (el > et ? el : et);
        } else {
            const int8_t* q = Q7().q[G] + 1024; const int ga = c - a, gb = c - b;
            int qa = q[ga], qb = q[gb], qd = q[d - b];
            s = ga ? (ga > 0 ? 1 : -1) : (gb < 0 ? -1 : 1);                 // PLT2 tiny sign
            if (MODE != 3 && s < 0) { qa = -qa; qb = -qb; qd = -qd; }          // mirror: the cell of -grads, symbol s * e
            const int cell = ((3 - qa) * 7 + qb + 3) * 7 + qd + 3;
            if (MODE == 2) cx = cell * 9 + (el > et ? el : et);
            else {
                int v = 0; if (y) { v = Zp[x - 1] > Zp[x] ? Zp[x - 1] : Zp[x]; v = v > Zp[x + 1] ? v : Zp[x + 1]; }
                const int lev = (v >= VT1) + (v >= VT2);
                cx = MODE == 1 ? cell * 3 + lev : llcomp_fse::kGroups[G][cell * 3 + lev];
            }
        }
    }
    LLC_AI void enc(Buf& B, RcEnc& E, BitW& Bw, int x, int y) {
        int pred, cx, s; ctx(x, y, pred, cx, s);
        int e = ((P[x] - pred + HALF) & MASK) - HALF;
        Ec[x] = S().bw8[e < 0 ? -e : e];
        if (MODE == 1 || MODE == 3) { int se = s * e; Zc[x] = uint16_t(se >= 0 ? 2 * se : -2 * se - 1); }
        uint16_t q = ENC31[G][s * e + 256]; int l = q & 31;
        if (MODE >= 4) if (FILE* f = dump_file()) { uint8_t r[4] = {uint8_t(cell), uint8_t(cell >> 8), uint8_t(l), uint8_t(G)}; fwrite(r, 1, 4, f); }
        Cdf& d = M->cdf[cx];
        E.put(B, d.c[l], d.c[l + 1] - d.c[l]);
        cdf_update(d, M->cnt[cx], l);
        if (int nb = RAW31[G][l]) Bw.put(B, q >> 5, nb);
    }
    LLC_AI void dec(const uint8_t* b, uint32_t& nx, RcDec& D, BitR& R, int x, int y) {
        int pred, cx, s; ctx(x, y, pred, cx, s);
        Cdf& d = M->cdf[cx];
        uint32_t r = D.range >> 15, v = D.code / r; v = v < 32767 ? v : 32767;
        int l = cdf_find(d, v);
        D.step(d.c[l], d.c[l + 1] - d.c[l], r, b, nx);
        cdf_update(d, M->cnt[cx], l);
        int a = LO31[G][l];
        if (int nb = RAW31[G][l]) a += int(R.get(b, nx, nb));
        Ec[x] = S().bw8[a];
        if (MODE == 1 || MODE == 3) { int se = SIGN31[G][l] * a; Zc[x] = uint16_t(se >= 0 ? 2 * se : -2 * se - 1); }
        int vv = pred + s * SIGN31[G][l] * a;
        P[x] = int16_t(KIND ? (((vv + 256) & 511) - 256) : (vv & 255));
    }
};

// ------------------------------------------------------------------ strips
struct Strips {
    int n, W, q, r;
    int x(int i) const { return n == 1 ? 0 : 64 * (i * q + std::min(i, r)); }
    int w(int i) const { return i + 1 < n ? 64 * (q + (i < r)) : W - x(i); }
};
inline Strips strips_for(int W, int req) {
    int n = req ? std::min(req, std::max(1, W / 64)) : std::max(1, W / 512), B = W / 64;
    return Strips{n, W, n > 1 ? B / n : 0, n > 1 ? B % n : 0};
}
template <class F> void parallel_for(int n, int threads, F f) {
    if (threads <= 1 || n <= 1) { for (int i = 0; i < n; ++i) f(i); return; }
    std::atomic<int> next{0}; std::vector<std::thread> th;
    for (int t = 0; t < std::min(threads, n); ++t) th.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); });
    for (auto& t : th) t.join();
}

// channel layout per pixel: CH = 1 (gray), 3 (RGB -> Y, U, V), 4 (RGBA -> Y, U, V, A)
template <int CH, int MODE>
std::vector<uint8_t> enc_strip(const uint8_t* img, int W, int x0, int w, int H) {
    Buf B; B.b.resize(size_t(w) * H * CH / 2 + 256);
    constexpr int NR = CH == 1 ? 1 : CH;                    // range coders
    RcEnc E[NR]; BitW Bw; for (auto& e : E) e.init(B);
    Model mY(nctx_of(MODE)), mC(nctx_of(MODE)), mA(nctx_of(MODE));
    if (MODE == 6) { mY.preset(0); mC.preset(1); mA.preset(0); }
    Chan<0, MODE> cy(&mY, w); Chan<1, MODE> cu(&mC, w), cv(&mC, w); Chan<0, MODE> ca(&mA, w);
    for (int y = 0; y < H; ++y) {
        cy.row(y); if (CH >= 3) { cu.row(y); cv.row(y); } if (CH == 4) ca.row(y);
        const uint8_t* p = img + (size_t(y) * W + x0) * CH;
        for (int x = 0; x < w; ++x) {
            if (CH == 1) cy.P[x] = p[x];
            else { int r = p[CH * x], g = p[CH * x + 1], b = p[CH * x + 2], v = r - g, u = b - g;
                   cy.P[x] = int16_t(g + ((u + v) >> 2)); cu.P[x] = int16_t(u); cv.P[x] = int16_t(v); if (CH == 4) ca.P[x] = p[CH * x + 3]; }
        }
        for (int x = 0; x < w; ++x) {
            cy.enc(B, E[0], Bw, x, y);
            if (CH >= 3) { cu.enc(B, E[1], Bw, x, y); cv.enc(B, E[2], Bw, x, y); }
            if (CH == 4) ca.enc(B, E[3], Bw, x, y);
        }
    }
    for (auto& e : E) e.flush(B);
    Bw.flush(B); B.b.resize(B.next); return std::move(B.b);
}
template <int CH, int MODE>
void dec_strip(const uint8_t* b, uint8_t* img, int W, int x0, int w, int H) {
    constexpr int NR = CH == 1 ? 1 : CH;
    RcDec D[NR]; BitR R; uint32_t n = 0; for (auto& d : D) d.init(b, n);
    Model mY(nctx_of(MODE)), mC(nctx_of(MODE)), mA(nctx_of(MODE));
    if (MODE == 6) { mY.preset(0); mC.preset(1); mA.preset(0); }
    Chan<0, MODE> cy(&mY, w); Chan<1, MODE> cu(&mC, w), cv(&mC, w); Chan<0, MODE> ca(&mA, w);
    for (int y = 0; y < H; ++y) {
        cy.row(y); if (CH >= 3) { cu.row(y); cv.row(y); } if (CH == 4) ca.row(y);
        for (int x = 0; x < w; ++x) {
            cy.dec(b, n, D[0], R, x, y);
            if (CH >= 3) { cu.dec(b, n, D[1], R, x, y); cv.dec(b, n, D[2], R, x, y); }
            if (CH == 4) ca.dec(b, n, D[3], R, x, y);
        }
        uint8_t* p = img + (size_t(y) * W + x0) * CH;
        for (int x = 0; x < w; ++x) {
            if (CH == 1) p[x] = uint8_t(cy.P[x]);
            else { int u = cu.P[x], v = cv.P[x], g = cy.P[x] - ((u + v) >> 2);
                   p[CH * x] = uint8_t(v + g); p[CH * x + 1] = uint8_t(g); p[CH * x + 2] = uint8_t(u + g); if (CH == 4) p[CH * x + 3] = uint8_t(ca.P[x]); }
        }
    }
}

template <int CH, int M> std::vector<uint8_t> enc_strip_m(const uint8_t* img, int W, int x0, int w, int H, int mode) {
    if constexpr (M + 1 < NMODES) { if (mode != M) return enc_strip_m<CH, M + 1>(img, W, x0, w, H, mode); }
    return enc_strip<CH, M>(img, W, x0, w, H);
}
template <int CH, int M> void dec_strip_m(const uint8_t* b, uint8_t* img, int W, int x0, int w, int H, int mode) {
    if constexpr (M + 1 < NMODES) { if (mode != M) return dec_strip_m<CH, M + 1>(b, img, W, x0, w, H, mode); }
    dec_strip<CH, M>(b, img, W, x0, w, H);
}
inline std::vector<uint8_t> compressImage(const std::vector<uint8_t>& img, int width, int height, int channels,
                                          int threads = int(std::thread::hardware_concurrency()), int nstrips = 0, int mode = 0) {
    if (mode < 0 || mode >= NMODES) throw std::invalid_argument("bad context mode");
    if (channels != 1 && channels != 3 && channels != 4) throw std::invalid_argument("channels must be 1, 3 or 4");
    if (width <= 0 || height <= 0 || width > 65535 || height > 65535 || img.size() != size_t(width) * height * channels) throw std::invalid_argument("bad image size");
    nstrips = std::max(0, std::min(nstrips, 255)); Strips st = strips_for(width, nstrips);
    std::vector<std::vector<uint8_t>> s(st.n);
    parallel_for(st.n, threads, [&](int i) {
        int x0 = st.x(i), w = st.w(i);
        s[i] = channels == 1 ? enc_strip_m<1, 0>(img.data(), width, x0, w, height, mode) : channels == 3 ? enc_strip_m<3, 0>(img.data(), width, x0, w, height, mode) : enc_strip_m<4, 0>(img.data(), width, x0, w, height, mode);
    });
    std::vector<uint8_t> o = {magic_rc, uint8_t(channels | mode << 4), uint8_t(width), uint8_t(width >> 8), uint8_t(height), uint8_t(height >> 8), uint8_t(nstrips)};
    for (auto& b : s) for (int k = 0; k < 4; ++k) o.push_back(uint8_t(b.size() >> (8 * k)));
    for (auto& b : s) o.insert(o.end(), b.begin(), b.end());
    return o;
}

struct RawImage { std::vector<uint8_t> pixels; uint16_t width; uint16_t height; uint8_t channels; };
inline RawImage decompressImage(const std::vector<uint8_t>& data, int threads = int(std::thread::hardware_concurrency())) {
    if (data.size() < 7 || data[0] != magic_rc) throw std::runtime_error("Invalid magic number");
    int ch = data[1] & 15, mode = data[1] >> 4, W = data[2] | data[3] << 8, H = data[4] | data[5] << 8;
    if ((ch != 1 && ch != 3 && ch != 4) || mode >= NMODES || W == 0 || H == 0) throw std::runtime_error("Invalid header");
    Strips st = strips_for(W, data[6]); size_t hdr = 7 + 4 * size_t(st.n); if (data.size() < hdr) throw std::runtime_error("Truncated");
    std::vector<size_t> off(st.n + 1, hdr);
    for (int i = 0; i < st.n; ++i) { size_t p = 7 + 4 * size_t(i); off[i + 1] = off[i] + (data[p] | data[p + 1] << 8 | data[p + 2] << 16 | size_t(data[p + 3]) << 24); }
    if (off.back() != data.size()) throw std::runtime_error("Bad strip lengths");
    std::vector<uint8_t> buf(data); buf.resize(data.size() + 16, 0);     // zero tail: no bounds checks in the hot loop
    RawImage out{std::vector<uint8_t>(size_t(W) * H * ch), uint16_t(W), uint16_t(H), uint8_t(ch)};
    parallel_for(st.n, threads, [&](int i) {
        const uint8_t* b = buf.data() + off[i]; int x0 = st.x(i), w = st.w(i);
        if (ch == 1) dec_strip_m<1, 0>(b, out.pixels.data(), W, x0, w, H, mode); else if (ch == 3) dec_strip_m<3, 0>(b, out.pixels.data(), W, x0, w, H, mode); else dec_strip_m<4, 0>(b, out.pixels.data(), W, x0, w, H, mode);
    });
    return out;
}
}  // namespace llcomp_rc
