#!/usr/bin/env python3
"""q7 x q7 context grid of MED residuals over a corpus, Monte Carlo, one A4 sheet: Y on top, U/V below.

usage:  python llcomp_q7grid.py <folder> [-n 100000] [-o q7grid.png] [--seed 1] [--max 0] [--folded]

For every image (.png/.ppm/.pnm, RGB) N random pixels are taken. Per pixel, per plane (RCT as llcomp:
V = R-G, U = B-G, Y = G + ((U+V) >> 2)): a = left, b = top, c = top-left (x = 0: a = c = b; y = 0: b = c = 0),
MED residual e = P - median(a, a+b-c, b), wrapped (Y mod 256, U/V mod 512). Axes: q7(c-a) vertical, q7(c-b)
horizontal, q7(x) = sign(x) * #(|x| >= 1, 4, 16) -> -16+ .. +16+. Each cell: share % and mean signed e.
Title: cost of folding the point-symmetric pairs (q1,q2) ~ (-q1,-q2) with e -> -e (zero-order entropy of the pooled
sample, tables over the whole corpus). --folded draws the grid after that mirror instead (pair shares, mean s*e).
Needs numpy, Pillow, matplotlib.
"""
import argparse, glob, os, sys
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LAB = ['−16+', '−4…15', '−1…3', '0', '+1…3', '+4…15', '+16+']

def q7(x): a = np.abs(x); return np.sign(x) * ((a >= 1).astype(np.int32) + (a >= 4) + (a >= 16))

def ent(z):
    if z.size == 0: return 0.0
    _, c = np.unique(z, return_counts=True); return float(-(c * np.log2(c / c.sum())).sum())

def sample(path, n, rng):
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
        out.append((int(n9), q7(c - a_), q7(c - b), e))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('-o', default='q7grid.png'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--folded', action='store_true'); A = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(A.folder, '*')) if f.lower().endswith(('.png', '.ppm', '.pnm')))
    if A.max: files = files[:A.max]
    if not files: sys.exit('no images')
    rng = np.random.default_rng(A.seed)
    cnt = np.zeros((2, 7, 7)); se = np.zeros((2, 7, 7)); zf = {0: [], 1: []}; kf = {0: [], 1: []}; zm = {0: [], 1: []}; km = {0: [], 1: []}
    fc = np.zeros((2, 7, 7)); fs = np.zeros((2, 7, 7))
    for i, f in enumerate(files):
        for t, q1, q2, e in sample(f, A.n, rng):
            idx = (q1 + 3) * 7 + (q2 + 3)
            cnt[t] += np.bincount(idx, minlength=49).reshape(7, 7); se[t] += np.bincount(idx, weights=e, minlength=49).reshape(7, 7)
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
    N_, S_ = (fc, fs) if A.folded else (cnt, se)
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
        ax.set_title(f'{nm}: {"после зеркала — доля пары %, средний s·e" if A.folded else "доля %, средний e"}   склейка зеркальных пар {cost[t]:+.2f}% энтропии',
                     loc='left', fontsize=10, color='#1f1f1d')
    fig.suptitle(f'q7 × q7, MED: {len(files)} изображений × {A.n} случайных пикселей  ({os.path.basename(os.path.normpath(A.folder))})', fontsize=11, color='#1f1f1d', x=0.02, ha='left')
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(A.o, dpi=300, facecolor='#ffffff')
    print(f'wrote {A.o}   fold cost: Y {cost[0]:+.3f}%  U/V {cost[1]:+.3f}%')

if __name__ == '__main__':
    main()
