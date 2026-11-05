#!/usr/bin/env python3
"""Sweep of q7 edges per plane type over a corpus, one pass over the sample cache.

usage:  python llcomp_q7edges.py <folder> [-n 100000] [--seed 1] [--max 0] [--plane uv|y|both] [--to 11,8] [-o q7edges.csv]
For every candidate edge set e1 < e2 < e3 (cells 0..e1-1 / e1..e2-1 / e2..e3-1 / e3+ on |c-a| and |c-b|, signed ->
7 x 7 = 49 sectors) it builds the |e| histograms per sector (same sampling and cache as llcomp_q7grid.py, see
llcomp_q7cache.py) and reports zero-order |e| bits per sample: all 49 sectors and after greedy merging (cheapest pair
first, as llcomp_q7similar.py) down to the sizes in --to. Candidates (default grids):
  U/V: e1 in 1..2, e2 in e1+1..6, e3 in e2+1..12
  Y:   e1 in 1..3, e2 in e1+1..8, e3 in {10, 12, 14, 16, 20, 24, 32}
1 4 16 is always added as the baseline. Prints the best 15 per plane type (by bits at the smallest --to size) and writes all rows to <o> (CSV).
Needs numpy, Pillow (and the cache from a previous run speeds it up a lot).
"""
import argparse, glob, os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llcomp_q7cache as Q
from llcomp_q7similar import agglo, H

M = 256
def grid(plane):
    g = _grid(plane); return g if (1, 4, 16) in g else [(1, 4, 16)] + g
def _grid(plane):
    if plane == 1: return [(a, b, c) for a in (1, 2) for b in range(a + 1, 7) for c in range(b + 1, 13)]
    return [(a, b, c) for a in (1, 2, 3) for b in range(a + 1, 9) for c in (10, 12, 14, 16, 20, 24, 32) if c > b]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--plane', default='both', choices=['uv', 'y', 'both']); ap.add_argument('--to', default='11,8')
    ap.add_argument('-o', default='q7edges.csv'); Q.add_args(ap); A = ap.parse_args()
    sizes = sorted({int(x) for x in A.to.split(',')}, reverse=True)
    files = sorted(f for f in glob.glob(os.path.join(A.folder, '*')) if f.lower().endswith(('.png', '.ppm', '.pnm')))
    if A.max: files = files[:A.max]
    if not files: sys.exit('no images')
    planes = {'uv': [1], 'y': [0], 'both': [0, 1]}[A.plane]
    C = {t: grid(t) for t in planes}; Hh = {t: np.zeros((len(C[t]), 49, M)) for t in planes}; N = {t: 0 for t in planes}
    print(f'{len(files)} files; candidates: ' + ', '.join(f'{"Y" if t == 0 else "U/V"} {len(C[t])}' for t in planes), flush=True)
    for i, f in enumerate(files):
        for t, ga, gb, e in Q.load(f, A.n, A.seed, A.cache, not A.no_cache):
            if t not in planes: continue
            ga = ga.astype(np.int32); gb = gb.astype(np.int32); ae = np.minimum(np.abs(e.astype(np.int32)), M - 1)
            aa, ab = np.minimum(np.abs(ga), 64), np.minimum(np.abs(gb), 64); sa, sb = np.sign(ga), np.sign(gb)
            # joint histogram over (|ga| capped, sign, |gb| capped, sign, |e|) is too big; per candidate bincount instead
            N[t] += ga.size
            for ci, E in enumerate(C[t]):
                lut = np.zeros(65, np.int32)
                for x in E: lut[x:] += 1
                q1 = sa * lut[aa]; q2 = sb * lut[ab]; k = (3 - q1) * 7 + (q2 + 3)
                Hh[t][ci] += np.bincount(k * M + ae, minlength=49 * M).reshape(49, M)
        if (i + 1) % 25 == 0 or i + 1 == len(files): print(f'  {i + 1}/{len(files)}', flush=True)
    rows = []
    for t in planes:
        nm = 'Y' if t == 0 else 'U/V'; res = []
        for ci, E in enumerate(C[t]):
            Hs = list(Hh[t][ci]); b49 = sum(H(h) for h in Hs); r = [b49 / N[t]]
            st = {g: c for g, c, _, _ in agglo(Hs, sizes)}; r += [(b49 + st[g]) / N[t] for g in sizes]; res.append((E, r))
        base = [r for E, r in res if E == (1, 4, 16)]; base = base[0] if base else res[0][1]
        res.sort(key=lambda x: x[1][-1])
        print(f'\n== {nm}: |e| bits per sample, best 15 by {sizes[-1]} groups (1 4 16 = ' + ' / '.join(f'{v:.4f}' for v in base) + ')')
        print('   edges        49        ' + '  '.join(f'{g:2d} groups' for g in sizes))
        for E, r in res[:15]:
            print(f'   {E[0]:2d} {E[1]:2d} {E[2]:2d}   ' + '  '.join(f'{v:.4f} ({100 * v / b - 100:+.2f}%)' for v, b in zip(r, base)))
        rows += [(nm, E, r) for E, r in res]
    with open(A.o, 'w', encoding='utf-8') as fh:
        fh.write('plane,e1,e2,e3,bits_49,' + ','.join(f'bits_{g}' for g in sizes) + '\n')
        for nm, E, r in rows: fh.write(f'{nm},{E[0]},{E[1]},{E[2]},' + ','.join(f'{v:.5f}' for v in r) + '\n')
    print(f'\nwrote {A.o}')

if __name__ == '__main__':
    main()
