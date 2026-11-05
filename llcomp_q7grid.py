#!/usr/bin/env python3
"""q7 x q7 context grid of MED residuals over a corpus, Monte Carlo, one A4 sheet: Y on top, U/V below.

usage:  python llcomp_q7grid.py <folder> [-n 100000] [-o q7grid.png] [--seed 1] [--max 0] [--kb 1.125]
--kb K: c-b is stretched by K before q7 (|x|*K rounded down; 1.125 = x + (x>>3)) to even the grid about its diagonal
writes two A4 PNGs: <o> (full grid, signed) and <o>_folded.png (after the mirror, no sign), and <o>.csv: per sector
1..49 (row by row from the top-left: q(c-a) +16+ .. -16+, q(c-b) -16+ .. +16+) its flat cell |h|, mirror sign, and
per plane type share %, mean e, mean |e|; and <o>_sectors.csv in the spreadsheet layout (Excel, decimal comma,
UTF-8 with BOM, sorted by the Y share): Сектор, s, q(c−a), q(c−b), Плоская ячейка |h|, Y %, U/V %, Y |e|, U/V |e|

For every image (.png/.ppm/.pnm, RGB) N random pixels are taken. Per pixel, per plane (RCT as llcomp:
V = R-G, U = B-G, Y = G + ((U+V) >> 2)): a = left, b = top, c = top-left (x = 0: a = c = b; y = 0: b = c = 0),
MED residual e = P - median(a, a+b-c, b), wrapped (Y mod 256, U/V mod 512). Axes: q7(c-a) vertical, q7(c-b)
horizontal, q7(x) = sign(x) * #(|x| >= 1, 4, 16) -> -16+ .. +16+. Each cell: share % and mean signed e.
Title: cost of folding the point-symmetric pairs (q1,q2) ~ (-q1,-q2) with e -> -e (zero-order entropy of the pooled
sample, tables over the whole corpus). The second sheet is the grid after that mirror (pair shares, mean s*e).
Samples are cached (see llcomp_q7cache.py: --cache, default ~/.cache/clic-q7; --no-cache).
Needs numpy, Pillow, matplotlib.
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llcomp_q7cache as Q
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LAB = ['−16+', '−4…15', '−1…3', '0', '+1…3', '+4…15', '+16+']

def q7(x): a = np.abs(x); return np.sign(x) * ((a >= 1).astype(np.int32) + (a >= 4) + (a >= 16))

def ent(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())

def sample(path, n, rng, kb=1.0):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.int32)
    H, W = a.shape[:2]
    ys = rng.integers(0, H, n); xs = rng.integers(0, W, n)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2)
    out = []
    for P, n9 in ((Y, False), (V, True), (U, True)):
        p = P[ys, xs]
        b = np.where(ys > 0, P[np.maximum(ys - 1, 0), xs], 0)
        a_ = np.where(xs > 0, P[ys, np.maximum(xs - 1, 0)], b)
        c = np.where((ys > 0) & (xs > 0), P[np.maximum(ys - 1, 0), np.maximum(xs - 1, 0)], b)
        c = np.where(xs == 0, b, c)
        e = p - np.clip(a_ + b - c, np.minimum(a_, b), np.maximum(a_, b))
        e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
        g = c - b; g = np.sign(g) * np.floor(np.abs(g) * kb + 1e-9).astype(np.int32) if kb != 1.0 else g
        out.append((int(n9), q7(c - a_), q7(g), e))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('-o', default='q7grid.png'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--kb', type=float, default=1.0); Q.add_args(ap)
    A = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(A.folder, '*')) if f.lower().endswith(('.png', '.ppm', '.pnm')))
    if A.max: files = files[:A.max]
    if not files: sys.exit('no images')
    rng = np.random.default_rng(A.seed)
    cnt = np.zeros((2, 7, 7)); se = np.zeros((2, 7, 7)); sa = np.zeros((2, 7, 7)); zf = {0: [], 1: []}; kf = {0: [], 1: []}; zm = {0: [], 1: []}; km = {0: [], 1: []}
    fc = np.zeros((2, 7, 7)); fs = np.zeros((2, 7, 7))
    for i, f in enumerate(files):
        for t, ga, gb, e in Q.load(f, A.n, A.seed, A.cache, not A.no_cache):
            ga = ga.astype(np.int32); gb = gb.astype(np.int32); e = e.astype(np.int32)
            if A.kb != 1.0: gb = np.sign(gb) * np.floor(np.abs(gb) * A.kb + 1e-9).astype(np.int32)
            q1, q2 = q7(ga), q7(gb)
            idx = (q1 + 3) * 7 + (q2 + 3)
            cnt[t] += np.bincount(idx, minlength=49).reshape(7, 7); se[t] += np.bincount(idx, weights=e, minlength=49).reshape(7, 7)
            sa[t] += np.bincount(idx, weights=np.abs(e), minlength=49).reshape(7, 7)
            h = q2 + 7 * q1; s = np.where(h < 0, -1, 1); fi = (s * q1 + 3) * 7 + (s * q2 + 3)
            fc[t] += np.bincount(fi, minlength=49).reshape(7, 7); fs[t] += np.bincount(fi, weights=s * e, minlength=49).reshape(7, 7)
            M = 511 if t else 255; H_ = 256 if t else 128
            zz = lambda v: np.where(v >= 0, 2 * v, -2 * v - 1)
            zf[t].append(zz(e)); kf[t].append(idx)
            em = ((s * e + H_) & M) - H_; zm[t].append(zz(em)); km[t].append(np.abs(h))
        if (i + 1) % 50 == 0 or i + 1 == len(files): print(f'  {i + 1}/{len(files)}', flush=True)
    cost = {}
    for t in (0, 1):
        z1, k1 = np.concatenate(zf[t]), np.concatenate(kf[t]); z2, k2 = np.concatenate(zm[t]), np.concatenate(km[t])
        h1 = sum(ent(z1[k1 == v]) for v in np.unique(k1)); h2 = sum(ent(z2[k2 == v]) for v in np.unique(k2))
        cost[t] = 100 * h2 / h1 - 100
    base, ext = os.path.splitext(A.o)
    with open(base + '.csv', 'w', encoding='utf-8') as fh:
        fh.write('sector,q_c_a,q_c_b,flat_cell,sign,Y_share_pct,Y_mean_e,Y_mean_abs_e,UV_share_pct,UV_mean_e,UV_mean_abs_e\n')
        for k in range(1, 50):
            r, col = (k - 1) // 7, (k - 1) % 7; i, j = 6 - r, col; q1, q2 = i - 3, j - 3; h = q2 + 7 * q1
            row = [str(k), LAB[i], LAB[j], str(abs(h)), '+' if h >= 0 else '-']
            for t in (0, 1):
                n_ = cnt[t, i, j]; row += [f'{100 * n_ / cnt[t].sum():.4f}', f'{se[t, i, j] / max(n_, 1):.4f}', f'{sa[t, i, j] / max(n_, 1):.4f}']
            fh.write(','.join(row) + '\n')
    print(f'wrote {base}.csv')
    rows = []
    for k in range(1, 50):
        r, col = (k - 1) // 7, (k - 1) % 7; i, j = 6 - r, col; q1, q2 = i - 3, j - 3; h = q2 + 7 * q1
        sh = [100 * cnt[t, i, j] / cnt[t].sum() for t in (0, 1)]; ma = [sa[t, i, j] / max(cnt[t, i, j], 1) for t in (0, 1)]
        rows.append((sh[0], k, '+' if h >= 0 else '−', LAB[i], LAB[j], abs(h), sh[1], ma[0], ma[1]))
    dc = lambda v, d: '"' + f'{v:.{d}f}'.replace('.', ',') + '"'
    with open(base + '_sectors.csv', 'w', encoding='utf-8-sig', newline='') as fh:
        fh.write('Сектор,s,q(c−a),q(c−b),Плоская ячейка |h|,"Y, %","U/V, %","Y, |e|","U/V, |e|"\r\n')
        for y, k, sg, la, lb, fl, uv, my, mu in sorted(rows, key=lambda t: (-round(t[0], 1), -t[0])):
            fh.write(f'{k},{sg},{la},{lb},{fl},{dc(y, 1)},{dc(uv, 1)},{dc(my, 2)},{dc(mu, 2)}\r\n')
    print(f'wrote {base}_sectors.csv')
    for folded, N_, S_, out in ((False, cnt, se, A.o), (True, fc, fs, base + '_folded' + (ext or '.png'))):
        draw(folded, N_, S_, cost, out, len(files), A)
    print(f'fold cost: Y {cost[0]:+.3f}%  U/V {cost[1]:+.3f}%')

def draw(folded, N_, S_, cost, out, nfiles, A):
    cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
    fig, axs = plt.subplots(2, 1, figsize=(8.27, 11.69), facecolor='#ffffff')
    for ax, (t, nm) in zip(axs, ((0, 'Y'), (1, 'U/V'))):
        sh = 100 * N_[t] / N_[t].sum(); me = S_[t] / np.maximum(N_[t], 1)
        vmax = max(1e-9, np.sort(sh.ravel())[-2]) if sh.max() > 2.5 * np.sort(sh.ravel())[-2] else sh.max()
        ax.imshow(np.where(N_[t] > 0, sh, np.nan), origin='lower', cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        for i in range(7):
            for j in range(7):
                if N_[t, i, j] == 0: continue
                ink = '#ffffff' if sh[i, j] > vmax * 0.55 else '#1f1f1d'
                ax.text(j, i + 0.15, f'{sh[i, j]:.0f}' if sh[i, j] >= 10 else f'{sh[i, j]:.1f}', ha='center', va='center', fontsize=10, weight='bold', color=ink)
                ax.text(j, i - 0.22, f'e {me[i, j]:+.1f}', ha='center', va='center', fontsize=7.5, color=ink)
        ax.plot([-0.5, 6.5], [-0.5, 6.5], color='#d8d7d0', lw=0.8, ls=':')
        ax.set_xticks(range(7)); ax.set_xticklabels(LAB, fontsize=8, color='#5e5d59'); ax.set_yticks(range(7)); ax.set_yticklabels(LAB, fontsize=8, color='#5e5d59')
        ax.set_xlabel('q(c−b)', color='#3d3d3a'); ax.set_ylabel('q(c−a)', color='#3d3d3a'); ax.tick_params(length=0)
        for s_ in ax.spines.values(): s_.set_color('#d8d7d0')
        ax.set_title(f'{nm}: {"после зеркала (без знака) — доля пары %, средний s·e" if folded else "со знаком — доля %, средний e"}   склейка зеркальных пар {cost[t]:+.2f}% энтропии',
                     loc='left', fontsize=10, color='#1f1f1d')
    fig.suptitle(f'q7 × q7, MED: {nfiles} изображений × {A.n} случайных пикселей  ({os.path.basename(os.path.normpath(A.folder))})', fontsize=11, color='#1f1f1d', x=0.02, ha='left')
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(out, dpi=300, facecolor='#ffffff'); plt.close(fig)
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
