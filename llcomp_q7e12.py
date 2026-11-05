#!/usr/bin/env python3
"""Third context axis from the neighbours' residuals: L = lg2(|E1| + |E2|) on top of the q7 x q7 sectors, Monte Carlo.

usage:  python llcomp_q7e12.py <folder> [-n 100000] [--seed 1] [--max 0] [--y 1,4,16] [--uv 1,4,16] [--lmax 8]
                               [--to 128,64,32,16,11,8] [-o q7e12]
E1 = MED residual of the left neighbour, E2 = of the top neighbour (wrapped, 0 outside the image), from the same cache
as the other q7 scripts (llcomp_q7cache.py; old caches without E1/E2 are rebuilt once). L = bit_length(|E1| + |E2|),
capped at --lmax (0 = both neighbours predicted exactly). Cells = 49 sectors (numbering as llcomp_q7grid.py) x L.
Per plane type (Y, U/V), zero-order |e| bits per sample: 49 sectors alone, 49 x L, and greedy merging of the 49 x L
cells (always the cheapest pair, merge cost = growth of the summed entropy) down to the sizes in --to, with the groups.
Writes <o>.txt (everything printed) and <o>_cells.csv (plane, sector, L, share %, mean |e|).
Needs numpy, Pillow.
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llcomp_q7cache as Q

M = 256
def qe(x, E): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in E)
def Hrows(h):
    """entropy (bits) of every row of h"""
    n = h.sum(-1, keepdims=True); p = np.where(h > 0, h / np.maximum(n, 1), 1); return -(h * np.log2(p)).sum(-1)

def agglo(hs, sizes):
    """greedy merging of histogram rows; returns {size: (total entropy, groups)}"""
    hs = hs.copy(); ids = [[i] for i in range(len(hs))]; ent = Hrows(hs); n = len(hs)
    D = np.full((n, n), np.inf)
    for i in range(n): D[i, i + 1:] = Hrows(hs[i] + hs[i + 1:]) - ent[i] - ent[i + 1:]
    alive = np.ones(n, bool); out = {}; k = n
    while k > min(sizes):
        i, j = np.unravel_index(np.argmin(D), D.shape)
        hs[i] += hs[j]; ent[i] = Hrows(hs[i][None])[0]; ids[i] += ids[j]; alive[j] = False
        D[j, :] = np.inf; D[:, j] = np.inf
        r = np.where(alive)[0]; r = r[r != i]; c = Hrows(hs[i][None] + hs[r]) - ent[i] - ent[r]
        lo = r < i; D[r[lo], i] = c[lo]; D[i, r[~lo]] = c[~lo]; k -= 1
        if k in sizes: out[k] = (float(ent[alive].sum()), [sorted(ids[q]) for q in np.where(alive)[0]])
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--y', default='1,4,16'); ap.add_argument('--uv', default='1,4,16'); ap.add_argument('--lmax', type=int, default=8)
    ap.add_argument('--to', default='128,64,32,16,11,8'); ap.add_argument('-o', default='q7e12'); Q.add_args(ap); A = ap.parse_args()
    sizes = sorted({int(x) for x in A.to.split(',')}, reverse=True); EG = {0: [int(x) for x in A.y.split(',')], 1: [int(x) for x in A.uv.split(',')]}
    files = sorted(f for f in glob.glob(os.path.join(A.folder, '*')) if f.lower().endswith(('.png', '.ppm', '.pnm')))
    if A.max: files = files[:A.max]
    if not files: sys.exit('no images')
    NL = A.lmax + 1; Hh = np.zeros((2, 49 * NL, M))
    for i, f in enumerate(files):
        for t, ga, gb, e, ea, eb in Q.load(f, A.n, A.seed, A.cache, not A.no_cache, e12=True):
            ga, gb = ga.astype(np.int32), gb.astype(np.int32); ae = np.minimum(np.abs(e.astype(np.int32)), M - 1)
            s = (3 - qe(ga, EG[t])) * 7 + (qe(gb, EG[t]) + 3)
            x = np.abs(ea.astype(np.int32)) + np.abs(eb.astype(np.int32)); L = np.minimum(np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) + 1, 0), A.lmax)
            Hh[t] += np.bincount((s * NL + L) * M + ae, minlength=49 * NL * M).reshape(49 * NL, M)
        if (i + 1) % 25 == 0 or i + 1 == len(files): print(f'  {i + 1}/{len(files)}', flush=True)
    txt = open(A.o + '.txt', 'w', encoding='utf-8')
    def say(x=''): print(x); txt.write(x + '\n')
    say(f'{len(files)} files x {A.n} px; L = bit_length(|E1|+|E2|) capped at {A.lmax}; edges Y {A.y}, U/V {A.uv}')
    cells = open(A.o + '_cells.csv', 'w', encoding='utf-8'); cells.write('plane,sector,L,share_pct,mean_abs_e\n')
    for t, nm in ((0, 'Y'), (1, 'U/V')):
        h = Hh[t]; N = h.sum(); hs = h.reshape(49, NL, M)
        b49 = Hrows(hs.sum(1)).sum() / N; full = Hrows(h).sum() / N; used = int((h.sum(1) > 0).sum())
        say(f'\n== {nm}: {int(N)} samples')
        say(f'   49 sectors:          {b49:.4f} |e| bits per sample')
        say(f'   49 x L ({used} used):  {full:.4f}  ({100 * full / b49 - 100:+.2f}% vs 49 sectors)')
        hl = hs.sum(0); nl = hl.sum(1)
        say('   L level: share % / mean |e|:  ' + '  '.join(f'L{l} {100 * nl[l] / N:.1f}/{(hl[l] * np.arange(M)).sum() / max(nl[l], 1):.2f}' for l in range(NL)))
        for s_ in range(49):
            for l in range(NL):
                c = hs[s_, l]; n_ = c.sum()
                if n_: cells.write(f'{nm},{s_ + 1},{l},{100 * n_ / N:.4f},{(c * np.arange(M)).sum() / n_:.4f}\n')
        idx = np.where(h.sum(1) > 0)[0]; res = agglo(h[idx], sizes)
        say('   greedy merging of the 49 x L cells:')
        for g in sizes:
            if g not in res: continue
            tot, groups = res[g]; say(f'     {g:3d} groups: {tot / N:.4f}  ({100 * tot / N / b49 - 100:+.2f}% vs 49 sectors)')
        g = min(sizes); tot, groups = res[g]; say(f'   the {g} groups (sector.L, by mean |e|):')
        gl = []
        for gr in groups:
            cid = idx[gr]; c = h[cid].sum(0); gl.append(((c * np.arange(M)).sum() / c.sum(), 100 * c.sum() / N, cid))
        for me, sh, cid in sorted(gl, key=lambda x: x[0]):
            Ls = sorted({int(c % NL) for c in cid}); say(f'     mean|e| {me:6.2f}  share {sh:5.2f}%  L {Ls[0]}..{Ls[-1]}  {len(cid)} cells: ' + ' '.join(f'{c // NL + 1}.{c % NL}' for c in cid))
    say(f'\nwrote {A.o}.txt, {A.o}_cells.csv')

if __name__ == '__main__':
    main()
