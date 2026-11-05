#!/usr/bin/env python3
"""Calibrate the axis scale so the q7 x q7 grid is symmetric about its diagonal (swap c-a <-> c-b), Monte Carlo.

usage:  python llcomp_q7calib.py <folder> [-n 100000] [--seed 1] [--max 0]
Same sampling and planes as llcomp_q7grid.py (RCT V = R-G, U = B-G, Y = G + ((U+V) >> 2); a = left, b = top,
c = top-left). c-b is multiplied by k before q7 (edges 1, 4, 16), k = 0.80 .. 1.60 step 0.01.
Asymmetry A(k) = 1/2 * sum over cells |share(q1,q2) - share(q2,q1)| in percent points (0 = perfectly even about the
diagonal 7-13-19-25-31-37-43; by the central mirror the other diagonal 1-9-...-49 follows). Printed per plane type:
A at k = 1, best k and A there, and A for cheap integer forms (x + (x>>3) = 1.125, x + (x>>2) = 1.25, ...).
Also the central-mirror asymmetry 1/2 * sum |share(q1,q2) - share(-q1,-q2)| for reference (a scale can't change it much).
Second part: Givens rotation of the (c-a, c-b) plane by -10..+10 deg (edges 0.5 3.5 15.5, so 0 deg = the integer grid),
asymmetry per angle and the share-weighted major-axis angle (45 deg = on the diagonal).
Samples are cached (llcomp_q7cache.py: --cache, default ~/.cache/clic-q7; --no-cache).
Needs numpy, Pillow.
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llcomp_q7cache as Q

def q7(x): a = np.abs(x); return np.sign(x) * ((a >= 1).astype(np.int32) + (a >= 4) + (a >= 16))

def sample(path, n, rng):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.int32)
    H, W = a.shape[:2]; ys = rng.integers(0, H, n); xs = rng.integers(0, W, n)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); out = []
    for P, t in ((Y, 0), (V, 1), (U, 1)):
        b = np.where(ys > 0, P[np.maximum(ys - 1, 0), xs], 0)
        a_ = np.where(xs > 0, P[ys, np.maximum(xs - 1, 0)], b)
        c = np.where((ys > 0) & (xs > 0), P[np.maximum(ys - 1, 0), np.maximum(xs - 1, 0)], b); c = np.where(xs == 0, b, c)
        out.append((t, c - a_, c - b))
    return out

def grid(ga, gb, k, intform=None):
    sb = intform(gb) if intform else np.sign(gb) * np.floor(np.abs(gb) * k + 1e-9)
    g = np.bincount((q7(ga) + 3) * 7 + q7(sb.astype(np.int64)) + 3, minlength=49).reshape(7, 7).astype(np.float64)
    return 100 * g / g.sum()
def qh(x): a = np.abs(x); return (np.sign(x) * ((a >= 0.5).astype(np.int64) + (a >= 3.5) + (a >= 15.5))).astype(np.int64)
def grid_rot(ga, gb, deg):
    r = np.radians(deg); x = np.cos(r) * ga - np.sin(r) * gb; y = np.sin(r) * ga + np.cos(r) * gb
    g = np.bincount((qh(x) + 3) * 7 + qh(y) + 3, minlength=49).reshape(7, 7).astype(np.float64); return 100 * g / g.sum()
def asym(S): return 0.5 * np.abs(S - S.T).sum()
def mirror(S): return 0.5 * np.abs(S - S[::-1, ::-1]).sum()

INT = [('x', lambda x: x), ('x + (x>>4)  1.0625', lambda x: np.sign(x) * (np.abs(x) + (np.abs(x) >> 4))),
       ('x + (x>>3)  1.125', lambda x: np.sign(x) * (np.abs(x) + (np.abs(x) >> 3))),
       ('x + (x>>2)  1.25', lambda x: np.sign(x) * (np.abs(x) + (np.abs(x) >> 2))),
       ('x + (x>>1)  1.5', lambda x: np.sign(x) * (np.abs(x) + (np.abs(x) >> 1)))]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--max', type=int, default=0); Q.add_args(ap); A = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(A.folder, '*')) if f.lower().endswith(('.png', '.ppm', '.pnm')))
    if A.max: files = files[:A.max]
    if not files: sys.exit('no images')
    rng = np.random.default_rng(A.seed); D = {0: ([], []), 1: ([], [])}
    for i, f in enumerate(files):
        for t, ga, gb, _ in Q.load(f, A.n, A.seed, A.cache, not A.no_cache): D[t][0].append(ga.astype(np.int64)); D[t][1].append(gb.astype(np.int64))
        if (i + 1) % 50 == 0 or i + 1 == len(files): print(f'  {i + 1}/{len(files)}', flush=True)
    ks = np.round(np.arange(0.80, 1.601, 0.01), 2)
    for t, nm in ((0, 'Y'), (1, 'U/V')):
        ga, gb = np.concatenate(D[t][0]), np.concatenate(D[t][1])
        S1 = grid(ga, gb, 1.0); Ak = np.array([asym(grid(ga, gb, k)) for k in ks]); kb = ks[np.argmin(Ak)]; Sb = grid(ga, gb, kb)
        print(f'\n{nm}: diagonal asymmetry at k=1: {asym(S1):.3f} pp   best k = {kb:.2f}: {Ak.min():.3f} pp   '
              f'(central mirror asymmetry {mirror(S1):.3f} -> {mirror(Sb):.3f} pp)')
        print('   k:   ' + '  '.join(f'{k:.2f}' for k in ks[::10]))
        print('   A:   ' + '  '.join(f'{a:.2f}' for a in Ak[::10]))
        for lab, fn in INT: print(f'   c-b -> {lab:20s} A = {asym(grid(ga, gb, 0, fn)):.3f} pp')
        C_ = np.cov(ga.astype(float), gb.astype(float)); th = 0.5 * np.degrees(np.arctan2(2 * C_[0, 1], C_[0, 0] - C_[1, 1]))
        print(f'   rotation: covariance major axis {th:.1f} deg (45 = diagonal, would need {45 - th:+.1f} deg)')
        for dg in (-10, -6, -4, -2, -1, 0, 1, 2, 4, 6, 10): print(f'     rotate {dg:+3d} deg: A = {asym(grid_rot(ga, gb, dg)):.3f} pp')
        print('   swapped pairs at best k (share %, k=1 -> best):')
        for (i1, j1) in ((4, 3), (5, 3), (6, 3), (5, 4), (6, 4), (6, 5), (4, 2), (5, 1)):
            print(f'     (q1,q2)=({i1 - 3:+d},{j1 - 3:+d}) vs ({j1 - 3:+d},{i1 - 3:+d}):  {S1[i1, j1]:.3f} / {S1[j1, i1]:.3f}  ->  {Sb[i1, j1]:.3f} / {Sb[j1, i1]:.3f}')

if __name__ == '__main__':
    main()
