#!/usr/bin/env python3
"""Mirror-sign rules for the HLC5 contexts on a corpus (repeat of llcomp backlog.md 'Знак зеркала' on q7 contexts).

usage:  python llcomp_q7sign.py <folder> [-n 100000] [--seed 1] [--max 0] [--y 1,3,10] [--uv 1,3,7]
                                [--vy 8,24] [--vuv 1,4] [--to 16,8] [-o q7sign]
Every rule gives a sign s per sample; the coded symbol is s*e (wrapped again: Y mod 256, U/V mod 512). Zero-order
entropy of the full symbol (sign included), tables per group: the context cells are merged greedily (cheapest pair
first, on the even files) for each rule separately, bits are measured on the odd files with those groups.
Two context sets: 2-D = 49 sectors q7(c-a) x q7(c-b) (HLC5 now), 4-D = sectors x q7(d-b) x 3 levels of |E1|+|E2|
(--vy / --vuv thresholds: v < t1, < t2, rest). E1/E2 = MED residuals of the left / top neighbour (before their own flip).
Rules (first non-zero term wins, else +1):
  none          s = +1
  tiny          c-a, c-b                 (HLC5 now; both terms 'c minus neighbour')
  tiny+d        c-a, c-b, c-d            (same orientation, d as the last tie-break)
  rc            c-a, b-c                 (llcomp_rc without d: mixed orientation)
  rc+d          c-a, b-c, b-d            (llcomp_rc)
  std           d-b, b-c, c-a            (llcomp Standard)
  res           E1, E2                   (signs of the neighbours' residuals)
  ressum        E1 + E2
  tiny|res      c-a, c-b, then E1, E2    (gradient sign; in the flat sector the residual signs)
  tiny|ressum   c-a, c-b, then E1 + E2
Writes <o>.txt. Needs numpy, Pillow and the shared cache (llcomp_q7cache.py, with d-b and E1/E2).
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llcomp_q7cache as Q
from llcomp_q7e12 import agglo, Hrows

def qe(x, E): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in E)
def first(*terms):
    s = np.zeros_like(terms[0])
    for t in terms: s = np.where(s != 0, s, np.sign(t))
    return np.where(s != 0, s, 1)
RULES = ['none', 'tiny', 'tiny+d', 'rc', 'rc+d', 'std', 'res', 'ressum', 'tiny|res', 'tiny|ressum']
def rule(name, ga, gb, gd, ea, eb):
    gcd = gb - gd                                                    # c - d = (c - b) - (d - b)
    return {'none': lambda: np.ones_like(ga), 'tiny': lambda: first(ga, gb), 'tiny+d': lambda: first(ga, gb, gcd),
            'rc': lambda: first(ga, -gb), 'rc+d': lambda: first(ga, -gb, -gd), 'std': lambda: first(gd, -gb, ga),
            'res': lambda: first(ea, eb), 'ressum': lambda: first(ea + eb), 'tiny|res': lambda: first(ga, gb, ea, eb),
            'tiny|ressum': lambda: first(ga, gb, ea + eb)}[name]()

def evaluate(tr, te, sizes):
    """greedy merging on train rows, entropy of test rows with the same groups (unseen cells -> group 0)"""
    idx = np.where(tr.sum(1) > 0)[0]; r = agglo(tr[idx].astype(np.float64), sizes); out = {}
    for g in sizes:
        lab = np.zeros(len(tr), np.int64)
        for k, gr in enumerate(r[g][1]): lab[idx[gr]] = k
        h = np.zeros((g, tr.shape[1])); np.add.at(h, lab, te.astype(np.float64)); out[g] = Hrows(h).sum()
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--y', default='1,3,10'); ap.add_argument('--uv', default='1,3,7')
    ap.add_argument('--vy', default='8,24'); ap.add_argument('--vuv', default='1,4'); ap.add_argument('--to', default='16,8')
    ap.add_argument('-o', default='q7sign'); Q.add_args(ap); A = ap.parse_args()
    sizes = sorted({int(x) for x in A.to.split(',')}, reverse=True)
    EG = {0: [int(x) for x in A.y.split(',')], 1: [int(x) for x in A.uv.split(',')]}
    VT = {0: [int(x) for x in A.vy.split(',')], 1: [int(x) for x in A.vuv.split(',')]}
    files = sorted(f for f in glob.glob(os.path.join(A.folder, '*')) if f.lower().endswith(('.png', '.ppm', '.pnm')))
    if A.max: files = files[:A.max]
    if len(files) < 2: sys.exit('need at least 2 images (train/test split)')
    MA = {0: 256, 1: 512}
    H2 = {(r, sp, t): np.zeros((49, MA[t]), np.float32) for r in RULES for sp in (0, 1) for t in (0, 1)}
    H4 = {(r, sp, t): np.zeros((1029, MA[t]), np.float32) for r in RULES for sp in (0, 1) for t in (0, 1)}
    for i, f in enumerate(files):
        sp = i & 1
        for (t, ga, gb, e, ea, eb), (_, _, _, _, gd) in zip(Q.load(f, A.n, A.seed, A.cache, not A.no_cache, e12=True),
                                                            Q.load(f, A.n, A.seed, A.cache, not A.no_cache, dd=True)):
            ga, gb, gd, e, ea, eb = (x.astype(np.int32) for x in (ga, gb, gd, e, ea, eb)); m = MA[t]
            sec = (3 - qe(ga, EG[t])) * 7 + (qe(gb, EG[t]) + 3); dc = qe(gd, EG[t]) + 3
            v = np.abs(ea) + np.abs(eb); lv = (v >= VT[t][0]).astype(np.int32) + (v >= VT[t][1])
            c4 = (sec * 7 + dc) * 3 + lv
            for r in RULES:
                s = rule(r, ga, gb, gd, ea, eb); z = ((s * e + m // 2) & (m - 1))            # wrapped s*e, offset to 0..m-1
                H2[r, sp, t] += np.bincount(sec * m + z, minlength=49 * m).reshape(49, m)
                H4[r, sp, t] += np.bincount(c4 * m + z, minlength=1029 * m).reshape(1029, m)
        if (i + 1) % 25 == 0 or i + 1 == len(files): print(f'  {i + 1}/{len(files)}', flush=True)
    txt = open(A.o + '.txt', 'w', encoding='utf-8')
    def say(x=''): print(x, flush=True); txt.write(x + '\n')
    say(f'{len(files)} files x {A.n} px (train = even, test = odd); sectors Y {A.y}, U/V {A.uv}; 4-D: + q7(d-b) + |E1|+|E2| '
        f'levels Y {A.vy}, U/V {A.vuv}; full symbol (s*e) entropy, bits per test sample')
    for t, nm in ((0, 'Y'), (1, 'U/V')):
        N = float(H2['none', 1, t].sum()); res = {}
        for r in RULES:
            res[r] = (evaluate(H2[r, 0, t], H2[r, 1, t], sizes), evaluate(H4[r, 0, t], H4[r, 1, t], sizes))
            print(f'   {nm} {r} done', flush=True)
        say(f'\n== {nm}: {int(N)} test samples')
        hdr = '   rule          ' + ''.join(f'  2-D/{g:<2d}  vs none  vs tiny ' for g in sizes) + ''.join(f'  4-D/{g:<2d}  vs none  vs tiny ' for g in sizes)
        say(hdr)
        for r in RULES:
            row = f'   {r:13s} '
            for k in (0, 1):
                for g in sizes:
                    b = res[r][k][g]; b0 = res['none'][k][g]; bt = res['tiny'][k][g]
                    row += f'  {b / N:.4f}  {100 * (b / b0 - 1):+6.2f}%  {100 * (b / bt - 1):+6.2f}%'
            say(row)
    say(f'\nwrote {A.o}.txt')

if __name__ == '__main__':
    main()
