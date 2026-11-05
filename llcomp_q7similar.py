#!/usr/bin/env python3
"""Homogeneity of the q7 x q7 sectors: |e| histograms per sector, which sectors look alike. Fast Monte Carlo.

usage:  python llcomp_q7similar.py <folder> [-n 100000] [-o q7similar.png] [--seed 1] [--max 0] [--to 25,16,11,8,5]
One pass: N random pixels per image (same sampling, RCT and MED as llcomp_q7grid.py), |e| histogram per sector 1..49
(numbering as there: row by row from the top-left, q(c-a) +16+ .. -16+, q(c-b) -16+ .. +16+), Y and U/V separately.
Similarity of two sectors = what merging them costs: n1*H1 + n2*H2 -> (n1+n2)*H(merged), zero-order entropy of |e|,
in % of the total |e| bits of that plane type (0 = identical shapes; it weights by population, as a coder would).
Output: <o> = A4 PNG, per plane type a 49 x 49 merge-cost map (sectors ordered by mean |e|) and the |e| curves;
<o>.txt = nearest neighbour of every sector and greedy agglomerative merging (always the cheapest pair) down to
the sizes in --to, with the groups and the cumulative cost.
Samples are cached (llcomp_q7cache.py: --cache, default ~/.cache/clic-q7; --no-cache).
Needs numpy, Pillow, matplotlib.
"""
import argparse, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llcomp_q7grid import q7
import llcomp_q7cache as Q

M = 256                                   # |e| bins (U/V |e| above 255 go to the last bin)
def H(h): c = h[h > 0]; return float(-(c * np.log2(c / c.sum())).sum()) if c.size else 0.0
def label(k): r, j = (k - 1) // 7, (k - 1) % 7; return k

def merge_matrix(Hs):
    n = len(Hs); e = np.array([H(h) for h in Hs]); D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n): D[i, j] = D[j, i] = H(Hs[i] + Hs[j]) - e[i] - e[j]
    return D, e

def agglo(Hs, sizes):
    groups = [[k + 1] for k in range(49)]; hs = [h.copy() for h in Hs]; ent = [H(h) for h in hs]; total = 0.0; out = []
    while len(groups) > min(sizes):
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                c = H(hs[i] + hs[j]) - ent[i] - ent[j]
                if best is None or c < best[0]: best = (c, i, j)
        c, i, j = best; total += c
        groups[i] = sorted(groups[i] + groups[j]); hs[i] = hs[i] + hs[j]; ent[i] = H(hs[i]); del groups[j], hs[j], ent[j]
        if len(groups) in sizes: out.append((len(groups), total, [g[:] for g in groups], [h.copy() for h in hs]))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('-o', default='q7similar.png'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--to', default='25,16,11,8,5'); Q.add_args(ap); A = ap.parse_args(); sizes = sorted({int(x) for x in A.to.split(',')}, reverse=True)
    files = sorted(f for f in glob.glob(os.path.join(A.folder, '*')) if f.lower().endswith(('.png', '.ppm', '.pnm')))
    if A.max: files = files[:A.max]
    if not files: sys.exit('no images')
    rng = np.random.default_rng(A.seed); Hh = np.zeros((2, 49, M))
    for i, f in enumerate(files):
        for t, ga, gb, e in Q.load(f, A.n, A.seed, A.cache, not A.no_cache):
            q1, q2 = q7(ga.astype(np.int32)), q7(gb.astype(np.int32)); e = e.astype(np.int32)
            k = (3 - q1) * 7 + (q2 + 3)                                     # sector - 1
            Hh[t] += np.bincount(k * M + np.minimum(np.abs(e), M - 1), minlength=49 * M).reshape(49, M)
        if (i + 1) % 50 == 0 or i + 1 == len(files): print(f'  {i + 1}/{len(files)}', flush=True)
    base, _ = os.path.splitext(A.o); txt = open(base + '.txt', 'w', encoding='utf-8')
    def say(s=''): print(s); txt.write(s + '\n')
    fig, ax = plt.subplots(2, 2, figsize=(8.27, 11.69), gridspec_kw={'width_ratios': [1.25, 1]})
    for t, nm in ((0, 'Y'), (1, 'U/V')):
        Hs = list(Hh[t]); D, e = merge_matrix(Hs); tot = e.sum(); P = 100 * D / tot
        n = Hh[t].sum(1); me = (Hh[t] * np.arange(M)).sum(1) / np.maximum(n, 1); order = np.argsort(me)
        say(f'== {nm}: {int(n.sum())} samples, |e| bits total {tot / n.sum():.3f} per sample')
        say('nearest sector (merge cost, % of the plane bits):')
        for k in order:
            P2 = P[k].copy(); P2[k] = np.inf; j = int(np.argmin(P2))
            say(f'  {k + 1:2d}  share {100 * n[k] / n.sum():5.2f}%  mean|e| {me[k]:6.2f}   ~ {j + 1:2d} ({P2[j]:.4f}%)')
        say('greedy merging (cheapest pair first):')
        for g, c, groups, hs in agglo(Hs, sizes):
            say(f'  {g:2d} groups: cost {100 * c / tot:+.3f}% of {nm} bits')
            gs = sorted(zip(groups, hs), key=lambda x: (x[1] * np.arange(M)).sum() / x[1].sum())
            for gr, h in gs: say(f'      mean|e| {(h * np.arange(M)).sum() / h.sum():6.2f}  share {100 * h.sum() / n.sum():5.2f}%  sectors {" ".join(map(str, gr))}')
        say()
        a = ax[t, 0]; im = a.imshow(np.log10(P[np.ix_(order, order)] + 1e-4), cmap='magma_r', vmin=-4, vmax=1)
        a.set_xticks(range(49)); a.set_yticks(range(49)); a.set_xticklabels([str(k + 1) for k in order], fontsize=4.5, rotation=90)
        a.set_yticklabels([str(k + 1) for k in order], fontsize=4.5); a.set_title(f'{nm}: merge cost log10(% of bits), sectors by mean |e|', fontsize=8)
        cb = fig.colorbar(im, ax=a, fraction=0.04); cb.ax.tick_params(labelsize=6)
        b = ax[t, 1]; cm = plt.get_cmap('viridis')
        for r, k in enumerate(order):
            h = Hh[t, k] / max(Hh[t, k].sum(), 1); b.semilogy(np.arange(64), np.maximum(h[:64], 1e-6), color=cm(r / 48), lw=0.7)
        b.set_title(f'{nm}: |e| histograms (colour = mean |e| rank)', fontsize=8); b.set_xlabel('|e|', fontsize=7); b.tick_params(labelsize=6)
    fig.suptitle(f'q7 x q7 sector homogeneity, {len(files)} images x {A.n} px', fontsize=10); fig.tight_layout(); fig.savefig(A.o, dpi=200)
    say(f'wrote {A.o}, {base}.txt')

if __name__ == '__main__':
    main()
