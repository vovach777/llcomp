# llcomp_signed_train.py — kLargeMap2 / kLargePreset2 for rc -x 9 / -x 10.
#
# Dump (6 bytes, append): LLC_DUMP=dump.bin LLC_DUMP_RAW=1  llcompc-rc -x 4 img out
#   cell u16 = signed cell index + 16637   (no fold, range 0..33274)
#   letter u8 = A31 letter of the raw residual e, not of s*e
#   plane u8, |e| u16
#
# The tables shipped with this commit were trained on the 24 Kodak colour
# images only (kodak_color_p6_full, the zip in plt2 branch ctx5 at
# hlc/research/data/kodak_color_p6_full.zip) and the bpp numbers were
# measured on those same 24 images. That is the mistake: in-sample.
# A Kodak-trained unfolded map was already tried (--two, tiny sign,
# rc modes 7/8) and not kept: gray -0.07%, colour photos +0.08%,
# big +0.34% against the FFV1 index sign.
#
# usage: python3 llcomp_signed_train.py dump.bin
import sys, numpy as np
import llcomp_large_cluster as C
C.NC, C.TWO, C.NL = 33275, True, 32
raw = np.fromfile(sys.argv[1], np.uint8).reshape(-1, 6)
cell = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8)
letter = raw[:, 2].astype(np.int32)
plane = raw[:, 3].astype(np.int32)
maps, hists = [], []
for g in (0, 1):
    sel = plane == g
    H = np.zeros((33275, 32), np.int64)
    np.add.at(H, (cell[sel], letter[sel]), 1)
    full, tot, rawb, ns, hk = C.cluster(H, 256)
    maps.append(full.astype(np.uint8))
    hists.append(hk)
    print('plane %d: seen %d  H(letter|cell) %.4f -> H(letter|cluster) %.4f' % (g, ns, rawb, tot), file=sys.stderr)
with open('llcomp_large_map2.h', 'w') as f:
    f.write('#pragma once\n#include <cstdint>\nnamespace llcomp_large {\nstatic const uint8_t kLargeMap2[2][33275] = {\n')
    for m in maps:
        f.write('  {' + ','.join(str(int(v)) for v in m) + '},\n')
    f.write('};\n}\n')
pres = np.zeros((2, 256, 32), np.int32)
for g in (0, 1):
    for k in range(256):
        h = hists[g][k][:31].astype(np.float64) + 0.5
        p = h / h.sum()
        w = 2 + np.floor(p * (32768 - 62)).astype(np.int32)
        w[int(np.argmax(w))] += 32768 - int(w.sum())
        c = np.concatenate([[0], np.cumsum(w)])
        pres[g, k] = c
with open('llcomp_large_preset2.h', 'w') as f:
    f.write('#pragma once\n#include <cstdint>\nnamespace llcomp_large {\nstatic const uint16_t kLargePreset2[2][256][32] = {\n')
    for g in (0, 1):
        f.write('{\n')
        for k in range(256):
            f.write('{' + ','.join(str(int(v)) for v in pres[g, k]) + '},\n')
        f.write('},\n')
    f.write('};\n}\n')
