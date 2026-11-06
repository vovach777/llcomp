#!/usr/bin/env python3
"""3-D context calibration: the up-right neighbour d as the third axis q7(d-b) on top of the q7(c-a) x q7(c-b) sectors.

usage:  python llcomp_q7d.py <folder> [-n 100000] [--seed 1] [--max 0] [--y 1,3,10] [--uv 1,3,7] [--to 16,11,8] [-o q7d]
        python llcomp_q7d.py <folder> --e3 [--dy 1,3,10] [--duv 1,3,7] ...   (4-D: + 3 error levels of |E1|+|E2|)
In the HLC5 wave t = x + k the neighbour d (y-1, x+1) is decoded in the same step as x; a skew t = x + 2k makes it one
step older, so d can enter the context. This script measures what it is worth, from the shared sample cache
(llcomp_q7cache.py; caches without d-b are rebuilt once). d borders: right edge d = b, top row d = b = 0.
Cells: 49 sectors (edges --y / --uv on c-a, c-b, numbering as llcomp_q7grid.py) x 7 signed q7 cells of d-b.
The edges of the d axis are swept per plane type (grids as llcomp_q7edges.py). Honest split: the cells are merged
greedily (cheapest pair first, as llcomp_q7similar.py) on the even files and the zero-order |e| bits per sample are
measured on the odd files with those groups; the baseline is the 49 sectors merged the same way to the same sizes.
Prints the best d edges per plane type, the gains and the groups of the best; writes <o>.txt and <o>.csv (all candidates).
--e3: no d sweep; d edges fixed (--dy / --duv, default = the sector edges); fourth axis = 3 unsigned levels of
v = |E1| + |E2| (residuals of the left and top neighbours): v < t1, t1 <= v < t2, v >= t2, the pair t1/t2 swept;
cells 49 x 7 x 3 = 1029 merged on train, measured on test; reports 2-D, 3-D (+d) and 4-D.
--zz (with --e3): v = zigzag(E1) + zigzag(E2) as the decoder holds them (no abs; ~ 2 (|E1| + |E2|), thresholds doubled).
Needs numpy, Pillow.
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llcomp_q7cache as Q
from llcomp_q7e12 import agglo, Hrows

M = 256; DC = 33; ND = 2 * DC + 1                                   # d-b clipped to [-33, 33] while accumulating
def qe(x, E): a = np.abs(x); return np.sign(x) * sum((a >= t).astype(np.int32) for t in E)
def grid(t):
    if t == 1: g = [(a, b, c) for a in (1, 2) for b in range(a + 1, 7) for c in range(b + 1, 13)]
    else: g = [(a, b, c) for a in (1, 2, 3) for b in range(a + 1, 9) for c in (10, 12, 14, 16, 20, 24, 32) if c > b]
    return g

def dmap(E):
    """raw clipped d-b value -> q7 cell 0..6"""
    v = np.arange(-DC, DC + 1); return qe(v, E) + 3

def apply_groups(groups, idx, train, test):
    """bits on test with the groups learned on train; cells unseen in train join the group of their biggest sibling"""
    lab = np.full(len(train), -1)
    for g, gr in enumerate(groups): lab[idx[gr]] = g
    miss = np.where((lab < 0) & (test.sum(1) > 0))[0]
    for c in miss:                                                    # same sector (cells are sector-major), other d
        s0 = (c // 7) * 7; sib = [x for x in range(s0, s0 + 7) if lab[x] >= 0]
        lab[c] = lab[max(sib, key=lambda x: train[x].sum())] if sib else 0
    h = np.zeros((len(groups), M)); np.add.at(h, lab[lab >= 0], test[lab >= 0]); return Hrows(h).sum()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--y', default='1,3,10'); ap.add_argument('--uv', default='1,3,7'); ap.add_argument('--to', default='16,11,8')
    ap.add_argument('--e3', action='store_true'); ap.add_argument('--zz', action='store_true'); ap.add_argument('--dy', default=''); ap.add_argument('--duv', default='')
    ap.add_argument('-o', default='q7d'); Q.add_args(ap); A = ap.parse_args()
    if A.e3: return main4(A)
    sizes = sorted({int(x) for x in A.to.split(',')}, reverse=True); EG = {0: [int(x) for x in A.y.split(',')], 1: [int(x) for x in A.uv.split(',')]}
    files = sorted(f for f in glob.glob(os.path.join(A.folder, '*')) if f.lower().endswith(('.png', '.ppm', '.pnm')))
    if A.max: files = files[:A.max]
    if len(files) < 2: sys.exit('need at least 2 images (train/test split)')
    J = np.zeros((2, 2, 49, ND, M))                                   # [split][plane type][sector][d-b][|e|]
    for i, f in enumerate(files):
        sp = i & 1
        for t, ga, gb, e, gd in Q.load(f, A.n, A.seed, A.cache, not A.no_cache, dd=True):
            ga, gb = ga.astype(np.int32), gb.astype(np.int32); ae = np.minimum(np.abs(e.astype(np.int32)), M - 1)
            s = (3 - qe(ga, EG[t])) * 7 + (qe(gb, EG[t]) + 3); dv = np.clip(gd.astype(np.int32), -DC, DC) + DC
            J[sp, t] += np.bincount((s * ND + dv) * M + ae, minlength=49 * ND * M).reshape(49, ND, M)
        if (i + 1) % 25 == 0 or i + 1 == len(files): print(f'  {i + 1}/{len(files)}', flush=True)
    txt = open(A.o + '.txt', 'w', encoding='utf-8'); csv = open(A.o + '.csv', 'w', encoding='utf-8')
    csv.write('plane,d1,d2,d3,' + ','.join(f'test_{g}' for g in sizes) + ',' + ','.join(f'gain_{g}_pct' for g in sizes) + '\n')
    def say(x=''): print(x, flush=True); txt.write(x + '\n')
    say(f'{len(files)} files x {A.n} px (train = even files, test = odd); sectors: edges Y {A.y}, U/V {A.uv}; '
        f'third axis q7(d-b); zero-order |e| bits per test sample')
    for t, nm in ((0, 'Y'), (1, 'U/V')):
        tr2 = J[0, t].sum(1); te2 = J[1, t].sum(1); N = te2.sum()     # 2-axis baseline: 49 sectors
        idx = np.where(tr2.sum(1) > 0)[0]; r2 = agglo(tr2[idx], sizes)
        base = {g: apply_groups(r2[g][1], idx, tr2, te2) / N for g in sizes if g in r2}
        say(f'\n== {nm}: {int(N)} test samples; 2 axes (49 sectors -> groups): ' + '  '.join(f'{g}: {base[g]:.4f}' for g in sizes if g in base))
        rows = []
        for E in grid(t):
            dm = dmap(E); tr = np.zeros((49, 7, M)); te = np.zeros((49, 7, M))
            np.add.at(tr, (slice(None), dm), J[0, t]); np.add.at(te, (slice(None), dm), J[1, t])
            tr = tr.reshape(343, M); te = te.reshape(343, M); idx = np.where(tr.sum(1) > 0)[0]; r3 = agglo(tr[idx], sizes)
            bits = {g: apply_groups(r3[g][1], idx, tr, te) / N for g in sizes if g in r3}
            rows.append((E, bits, r3, idx))
            csv.write(f'{nm},{E[0]},{E[1]},{E[2]},' + ','.join(f'{bits[g]:.5f}' for g in sizes) + ',' + ','.join(f'{100 * (bits[g] / base[g] - 1):.3f}' for g in sizes) + '\n')
        g0 = min(sizes); rows.sort(key=lambda r: r[1][g0])
        say(f'   best d edges (by test bits at {g0} groups); gain vs 2 axes at the same group count:')
        for E, bits, _, _ in rows[:12]:
            say(f'     d {E[0]:2d} {E[1]:2d} {E[2]:2d}: ' + '  '.join(f'{g}: {bits[g]:.4f} ({100 * (bits[g] / base[g] - 1):+.2f}%)' for g in sizes))
        same = [r for r in rows if list(r[0]) == EG[t]]
        if same: say(f'     (same edges as the sector axes, {" ".join(map(str, EG[t]))}: ' + '  '.join(f'{g}: {100 * (same[0][1][g] / base[g] - 1):+.2f}%' for g in sizes) + ')')
        E, bits, r3, idx = rows[0]; tr = np.zeros((49, 7, M)); np.add.at(tr, (slice(None), dmap(E)), J[0, t]); tr = tr.reshape(343, M)
        say(f'   the {g0} groups of the best (sector.dcell, dcell 0..6 = q(d-b) -3..+3), by mean |e| on train:')
        gl = []
        for gr in r3[g0][1]:
            cid = idx[gr]; c = tr[cid].sum(0); gl.append(((c * np.arange(M)).sum() / c.sum(), 100 * c.sum() / tr.sum(), cid))
        for me, sh, cid in sorted(gl, key=lambda x: x[0]):
            say(f'     mean|e| {me:6.2f}  share {sh:5.2f}%  {len(cid)} cells: ' + ' '.join(f'{c // 7 + 1}.{c % 7}' for c in cid))
    say(f'\nwrote {A.o}.txt, {A.o}.csv')

def main4(A):
    """4-D: sectors x q7(d-b) x 3 unsigned error levels of v = |E1| + |E2| (v < t1, t1 <= v < t2, v >= t2), t1/t2 swept"""
    sizes = sorted({int(x) for x in A.to.split(',')}, reverse=True); EG = {0: [int(x) for x in A.y.split(',')], 1: [int(x) for x in A.uv.split(',')]}
    ED = {0: [int(x) for x in (A.dy or A.y).split(',')], 1: [int(x) for x in (A.duv or A.uv).split(',')]}; VC = 128 if A.zz else 64
    files = sorted(f for f in glob.glob(os.path.join(A.folder, '*')) if f.lower().endswith(('.png', '.ppm', '.pnm')))
    if A.max: files = files[:A.max]
    if len(files) < 2: sys.exit('need at least 2 images (train/test split)')
    J = np.zeros((2, 2, 343, VC + 1, M), np.float32)                # v clipped to 0..64 while accumulating
    for i, f in enumerate(files):
        sp = i & 1
        for (t, ga, gb, e, ea, eb), (_, _, _, _, gd) in zip(Q.load(f, A.n, A.seed, A.cache, not A.no_cache, e12=True), Q.load(f, A.n, A.seed, A.cache, not A.no_cache, dd=True)):
            ga, gb, gd = ga.astype(np.int32), gb.astype(np.int32), gd.astype(np.int32); ae = np.minimum(np.abs(e.astype(np.int32)), M - 1)
            s = (3 - qe(ga, EG[t])) * 7 + (qe(gb, EG[t]) + 3); dc = qe(gd, ED[t]) + 3
            ea, eb = ea.astype(np.int32), eb.astype(np.int32)
            v = np.minimum((np.where(ea >= 0, 2 * ea, -2 * ea - 1) + np.where(eb >= 0, 2 * eb, -2 * eb - 1)) if A.zz else (np.abs(ea) + np.abs(eb)), VC)
            J[sp, t] += np.bincount(((s * 7 + dc) * (VC + 1) + v) * M + ae, minlength=343 * (VC + 1) * M).reshape(343, VC + 1, M)
        if (i + 1) % 25 == 0 or i + 1 == len(files): print(f'  {i + 1}/{len(files)}', flush=True)
    txt = open(A.o + '.txt', 'w', encoding='utf-8'); csv = open(A.o + '.csv', 'w', encoding='utf-8'); csv.write('plane,t1,t2,' + ','.join(f'gain_{g}_pct' for g in sizes) + '\n')
    def say(x=''): print(x, flush=True); txt.write(x + '\n')
    say(f'{len(files)} files x {A.n} px (train = even, test = odd); sectors Y {A.y}, U/V {A.uv}; d edges Y {ED[0]}, U/V {ED[1]}; '
        f'error axis: 3 levels of v = ' + ('zz(E1)+zz(E2) (zigzag)' if A.zz else '|E1|+|E2|') + ' (no sign); |e| bits per test sample')
    def ev(tr, te, sib):
        tr = tr.astype(np.float64); te = te.astype(np.float64)
        idx = np.where(tr.sum(1) > 0)[0]; r = agglo(tr[idx], sizes); out = {}
        for g in sizes:
            lab = np.full(len(tr), -1)
            for k, gr in enumerate(r[g][1]): lab[idx[gr]] = k
            for c in np.where((lab < 0) & (te.sum(1) > 0))[0]:
                s0 = (c // sib) * sib; cand = [x for x in range(s0, s0 + sib) if lab[x] >= 0]
                lab[c] = lab[max(cand, key=lambda x: tr[x].sum())] if cand else 0
            h = np.zeros((g, M)); np.add.at(h, lab[lab >= 0], te[lab >= 0]); out[g] = Hrows(h).sum()
        return out, r
    T1 = (1, 2, 3, 4, 6, 8, 12); T2 = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
    if A.zz: T1 = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24); T2 = (2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96)
    for t, nm in ((0, 'Y'), (1, 'U/V')):
        tr4, te4 = J[0, t], J[1, t]; N = float(te4.sum())
        tr3 = tr4.sum(1); te3 = te4.sum(1); tr2 = tr3.reshape(49, 7, M).sum(1); te2 = te3.reshape(49, 7, M).sum(1)
        b2, _ = ev(tr2, te2, 1); b3, _ = ev(tr3, te3, 7)
        say(f'\n== {nm}: {int(N)} test samples;  2-D: ' + '  '.join(f'{g}: {b2[g] / N:.4f}' for g in sizes) +
            ';  3-D (+d): ' + '  '.join(f'{g}: {100 * (b3[g] / b2[g] - 1):+.2f}%' for g in sizes))
        rows = []
        for t1 in T1:
            for t2 in T2:
                if t2 <= t1: continue
                lv = np.minimum((np.arange(VC + 1) >= t1).astype(int) + (np.arange(VC + 1) >= t2), 2)
                tr = np.zeros((343, 3, M), np.float32); te = np.zeros((343, 3, M), np.float32)
                for l in range(3): tr[:, l] = tr4[:, lv == l].sum(1); te[:, l] = te4[:, lv == l].sum(1)
                b4, r4 = ev(tr.reshape(-1, M), te.reshape(-1, M), 3); rows.append((t1, t2, b4))
                csv.write(f'{nm},{t1},{t2},' + ','.join(f'{100 * (b4[g] / b2[g] - 1):.3f}' for g in sizes) + '\n'); csv.flush()
                print(f'   t1 {t1:2d} t2 {t2:2d}: ' + '  '.join(f'{g}: {100 * (b4[g] / b2[g] - 1):+.2f}%' for g in sizes), flush=True)
        g0 = min(sizes); rows.sort(key=lambda r: r[2][g0])
        say(f'   best error thresholds (4-D = sectors x d x 3 error levels), gain vs 2-D at the same group count:')
        for t1, t2, b4 in rows[:8]:
            say(f'     v < {t1:2d} | < {t2:2d} | rest:  ' + '  '.join(f'{g}: {b4[g] / N:.4f} ({100 * (b4[g] / b2[g] - 1):+.2f}%)' for g in sizes))
    say(f'\nwrote {A.o}.txt, {A.o}.csv')

if __name__ == '__main__':
    main()
