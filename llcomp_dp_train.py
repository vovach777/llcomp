# Обучение -x 11. Знак вшит в уровень, остаток не инвертируется.
#
# Порог каждой из 5 осей ищется динамическим программированием по букве A31,
# отдельно для яркости и хромы. Потом произведение уровней сливается в корзины,
# и на каждую корзину пишется пресет. На выходе три заголовка, их подхватывает
# сборка, пересобирать кодек:
#
#   python3 llcomp_dp_train.py /path/to/clic --ky 256 --kuv 256
#   g++ -O3 -std=c++17 -pthread llcompc_rc.cpp -o llcompc-rc
#   g++ -O3 -std=c++17 -pthread llcompd_rc.cpp -o llcompd-rc
#
# Каталог или список файлов. PPM/PNM читаются сами, png/jpg — через ffmpeg.
# В памяти одна картинка. Гистограммы постоянного размера (ось 511, ячейка 33275),
# слияние идёт по ячейкам, не по пикселям. 1 ГБ картинок — это проход по пикселям
# и одно слияние, не гигабайты состояния.
#
# Число уровней по умолчанию 11,11,11,5,5: на Kodak срезание уровней
# увеличивало длину совместной ячейки. --levels меняет бюджет.
# --ky и --kuv не обязаны быть степенью двойки и не обязаны совпадать.
# Больше 256 нельзя: номер корзины — байт.
#
# Заголовки, которые лежат в репозитории, обучены на 24 Kodak
# (kodak_color_p6_full). На них же снят размер. Это не CLIC.
# Пока не перегенерируешь на своём корпусе, -x 11 — это Kodak.
import argparse, glob, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llcomp_q5_dp import read_ppm, planes, med_grad, residual, letter_of, dp_edges, apply_edges

def code_len(h):
    n = h.sum(-1, keepdims=True)
    good = h > 0
    return np.where(good, h * np.log2(np.where(good, n / np.maximum(h, 1), 1)), 0).sum(-1)

def files_of(args):
    out = []
    for a in args:
        if os.path.isdir(a):
            for dp, _, names in os.walk(a):
                for n in sorted(names):
                    if n.lower().endswith(('.ppm', '.pnm', '.png', '.jpg', '.jpeg')):
                        out.append(os.path.join(dp, n))
        else:
            out.extend(sorted(glob.glob(a)))
    return out

def load_rgb(path):
    if path.lower().endswith(('.ppm', '.pnm')):
        rgb = read_ppm(path)
        return rgb[:, :, :3] if rgb.ndim == 3 and rgb.shape[2] >= 3 else None
    wh = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path])
    w, h = [int(x) for x in wh.decode().replace(',', ' ').split()[:2]]
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)

def samples(rgb):
    ys = planes(rgb)
    for kind, P in ((0, ys[0]), (1, ys[1]), (1, ys[2])):
        gs, pred = med_grad(P, 128 if kind == 0 else 0)
        let = letter_of(residual(P, pred, kind), 0 if kind == 0 else 1).ravel()
        yield (0 if kind == 0 else 1), gs, let

def fit_edges(files, levels):
    H = np.zeros((2, 5, 511, 31), np.int64)
    for i, f in enumerate(files, 1):
        rgb = load_rgb(f)
        if rgb is None: continue
        for pl, gs, let in samples(rgb):
            for ax, g in enumerate(gs):
                gg = np.clip(g.astype(np.int32) + 255, 0, 510)
                np.add.at(H[pl, ax], (gg.ravel(), let), 1)
        if i % 20 == 0: print('edges', i, '/', len(files), file=sys.stderr)
    edges = []
    for pl in (0, 1):
        pe = []
        for ax, B in enumerate(levels):
            got = dp_edges(H[pl, ax], [B])
            pe.append(got[B][0])
            print('plane %d axis %d  B %d  H %.4f' % (pl, ax, B, got[B][1]), file=sys.stderr)
        edges.append(pe)
    return edges

def cell_hist(files, edges, levels):
    H = np.zeros((2, 33275, 31), np.int64)
    base = levels
    for i, f in enumerate(files, 1):
        rgb = load_rgb(f)
        if rgb is None: continue
        for pl, gs, let in samples(rgb):
            bins = [apply_edges(gs[ax], edges[pl][ax]).ravel() for ax in range(5)]
            key = bins[0].astype(np.int32); mul = base[0]
            for ax in range(1, 5):
                key = key + bins[ax] * mul; mul *= base[ax]
            np.add.at(H[pl], (key, let), 1)
        if i % 20 == 0: print('cells', i, '/', len(files), file=sys.stderr)
    return H

def baskets(h, K):
    seen = np.nonzero(h.sum(1) > 0)[0]
    X = h[seen].astype(np.float64)
    m = len(seen); NL = 31
    def lloyd(lab, k, it):
        for _ in range(it):
            Q = np.zeros((k, NL)); np.add.at(Q, lab, X)
            Q = (Q + 0.5) / (Q + 0.5).sum(1, keepdims=True)
            lab = np.argmin(-(X @ np.log2(Q).T), 1)
        return lab
    M0 = max(4 * K, 1024)
    if M0 > m: M0 = m
    top = np.argsort(-X.sum(1))[:M0]
    lab0 = np.zeros(m, int); lab0[top] = np.arange(M0)
    Q = (X[top] + 0.5) / (X[top] + 0.5).sum(1, keepdims=True)
    lab0 = np.argmin(-(X @ np.log2(Q).T), 1)
    lab0 = lloyd(lab0, M0, 8)
    used = np.unique(lab0); remap = {c: t for t, c in enumerate(used)}
    lab0 = np.array([remap[c] for c in lab0])
    Xc = np.zeros((len(used), NL)); np.add.at(Xc, lab0, X)
    Cmat = Xc.copy(); m2 = len(Xc); alive = np.ones(m2, bool); cl = code_len(Cmat); grp = np.arange(m2)
    cost = np.full((m2, m2), np.inf)
    def row_cost(i):
        return code_len(Cmat + Cmat[i]) - cl - cl[i]
    for i in range(m2):
        r = row_cost(i); r[i] = np.inf; cost[i] = r
    for step in range(m2 - K):
        k = int(np.argmin(cost)); i, j = divmod(k, m2)
        Cmat[i] += Cmat[j]; alive[j] = False; grp[grp == j] = i
        cl[i] = code_len(Cmat[i:i + 1])[0]
        cost[j, :] = np.inf; cost[:, j] = np.inf
        r = row_cost(i); r[~alive] = np.inf; r[i] = np.inf; cost[i] = r; cost[:, i] = r
    ids = {c: t for t, c in enumerate(sorted(set(grp[alive])))}
    lab = np.array([ids[grp[c]] for c in lab0])
    lab = lloyd(lab, K, 5)
    full = np.zeros(33275, np.uint8); full[seen] = lab.astype(np.uint8)
    Hk = np.zeros((256, 31), np.float64); np.add.at(Hk, lab, X)
    pres = np.zeros((256, 32), np.int32)
    for k in range(256):
        hh = Hk[k] + 0.5; p = hh / hh.sum()
        w = 2 + np.floor(p * (32768 - 62)).astype(np.int32)
        w[int(np.argmax(w))] += 32768 - int(w.sum())
        pres[k] = np.concatenate([[0], np.cumsum(w)])
    bits = code_len(Hk[:K]).sum() / X.sum()
    return full, pres, bits, len(seen)

def write_headers(out, edges, levels, maps, pres, note):
    def cuts(e):
        return [int(i - 255) for i in e[1:]]
    with open(os.path.join(out, 'llcomp_dp_cuts.h'), 'w') as f:
        f.write('#pragma once\n// %s\n' % note)
        for pl, tag in ((0, 'Y'), (1, 'UV')):
            f.write('static const int16_t DP%s[5][10] = {\n' % tag)
            for e in edges[pl]:
                c = cuts(e); c += [-32768] * (10 - len(c))
                f.write('  {' + ','.join(str(v) for v in c) + '},\n')
            f.write('};\nstatic const int DPN%s[5] = {%s};\n' % (tag, ','.join(str(len(cuts(e))) for e in edges[pl])))
    with open(os.path.join(out, 'llcomp_dp_map.h'), 'w') as f:
        f.write('#pragma once\n#include <cstdint>\n// %s\nnamespace llcomp_dp {\nstatic const uint8_t kDpMap[2][33275] = {\n' % note)
        for m in maps:
            f.write('  {' + ','.join(str(int(v)) for v in m) + '},\n')
        f.write('};\n}\n')
    with open(os.path.join(out, 'llcomp_dp_preset.h'), 'w') as f:
        f.write('#pragma once\n#include <cstdint>\n// %s\nnamespace llcomp_dp {\nstatic const uint16_t kDpPreset[2][256][32] = {\n' % note)
        for p in pres:
            f.write('{\n')
            for k in range(256):
                f.write('{' + ','.join(str(int(v)) for v in p[k]) + '},\n')
            f.write('},\n')
        f.write('};\n}\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--ky', type=int, default=256)
    ap.add_argument('--kuv', type=int, default=256)
    ap.add_argument('--levels', default='11,11,11,5,5')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    if not (1 <= a.ky <= 256 and 1 <= a.kuv <= 256):
        sys.exit('ky and kuv must be 1..256')
    levels = [int(v) for v in a.levels.split(',')]
    if len(levels) != 5 or any(not 1 <= b <= 15 for b in levels):
        sys.exit('levels: five integers, each 1..15')
    files = files_of(a.images)
    if not files: sys.exit('no images')
    print('%d images' % len(files), file=sys.stderr)
    edges = fit_edges(files, levels)
    H = cell_hist(files, edges, levels)
    maps, press = [], []
    for pl, K, name in ((0, a.ky, 'Y'), (1, a.kuv, 'UV')):
        m, p, bits, seen = baskets(H[pl], K)
        print('%s K %d  seen %d  H(letter|basket) %.4f' % (name, K, seen, bits), file=sys.stderr)
        maps.append(m); press.append(p)
    note = 'llcomp_dp_train.py ky=%d kuv=%d levels=%s  %d files' % (a.ky, a.kuv, a.levels, len(files))
    write_headers(a.out, edges, levels, maps, press, note)
    print(note, file=sys.stderr)

if __name__ == '__main__':
    main()
