# Per-axis DP quantizers for the 5 signed gradients, then the product cell.
# Sign stays in the bin: bins are contiguous pieces of the signed axis, no mirror.
# Kodak colour P6 only. Prints H(letter|axis) and H(letter|product).
import glob, os, sys, numpy as np

LO = np.array([
    [0,1,2,3,4,5,6,7,8,12,16,24,32,48,64,128],
    [0,1,2,3,4,5,6,7,8,10,12,14,16,24,32,64]], np.int32)
RAW = np.array([
    [0,0,0,0,0,0,0,0,2,2,3,3,4,4,6,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,3,3,5,8]], np.int32)

def letter_of(e, g):
    mag = np.abs(e)
    out = np.zeros(e.shape, np.int16)
    for i in range(1, 16):
        lo, span = int(LO[g, i]), 1 << int(RAW[g, i])
        if span == 1: span = 1
        hit = (mag >= lo) & (mag < lo + (span if RAW[g, i] else 1))
        out = np.where(hit, i, out)
    neg = e < 0
    return np.where(e == 0, 0, np.where(neg, out + 15, out)).astype(np.int16)

def read_ppm(path):
    with open(path, 'rb') as f:
        if f.readline().strip() != b'P6': raise SystemExit(path)
        line = f.readline()
        while line.startswith(b'#'): line = f.readline()
        w, h = map(int, line.split())
        f.readline()
        rgb = np.frombuffer(f.read(), np.uint8).reshape(h, w, 3)
    return rgb

def planes(rgb):
    r, g, b = rgb[:,:,0].astype(np.int16), rgb[:,:,1].astype(np.int16), rgb[:,:,2].astype(np.int16)
    v, u = r - g, b - g
    y = g + ((u + v) >> 2)
    return y, u, v

def med_grad(P, mid):
    a = np.empty_like(P); b = np.empty_like(P); c = np.empty_like(P); d = np.empty_like(P)
    ll = np.empty_like(P); tt = np.empty_like(P)
    a[:, 1:] = P[:, :-1]
    a[0, 0] = mid
    b[1:, :] = P[:-1, :]
    a[1:, 0] = b[1:, 0]
    b[0, :] = a[0, :]
    c[0, :] = a[0, :]
    c[1:, 0] = b[1:, 0]
    c[1:, 1:] = P[:-1, :-1]
    d[0, :] = a[0, :]
    d[1:, :-1] = P[:-1, 1:]
    d[1:, -1] = b[1:, -1]
    ll[:, 2:] = P[:, :-2]
    ll[:, 1] = a[:, 1]
    ll[:, 0] = a[:, 0]
    tt[2:, :] = P[:-2, :]
    tt[0, :] = b[0, :]
    tt[1, :] = b[1, :]
    mn = np.minimum(a, b); mx = np.maximum(a, b)
    pred = np.clip(a + b - c, mn, mx)
    return (a - c, c - b, b - d, ll - a, tt - b), pred

def residual(P, pred, kind):
    half, mask = (128, 255) if kind == 0 else (256, 511)
    return ((P - pred + half) & mask) - half

NAMES = ['a-c', 'c-b', 'b-d', 'll-a', 'tt-b']

def H_of(h):
    # h (..., 31)
    n = h.sum(-1)
    p = np.divide(h, np.maximum(n, 1)[..., None], where=n[..., None] > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        hh = np.where(h > 0, -(p * np.log2(np.where(p > 0, p, 1))), 0).sum(-1)
    return hh, n

def dp_edges(hist, Bs):
    # hist[511, 31], index 0 = gradient -255. returns edges (left index) per B
    N = hist.shape[0]
    pref = np.zeros((N + 1, 31), np.float64)
    pref[1:] = np.cumsum(hist, 0)
    cost = np.full((N, N + 1), np.inf)
    for i in range(N):
        h = pref[i + 1:] - pref[i]
        n = h.sum(1)
        good = n > 0
        p = np.zeros_like(h)
        p[good] = h[good] / n[good, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = np.where(h > 0, -(p * np.log2(np.where(p > 0, p, 1))), 0).sum(1)
        cost[i, i + 1:] = np.where(good, n * ent, 0)
    Bmax = max(Bs)
    dp = np.full((Bmax + 1, N + 1), np.inf)
    prv = np.full((Bmax + 1, N + 1), -1, np.int16)
    dp[0, 0] = 0
    for b in range(1, Bmax + 1):
        for i in range(b, N + 1):
            c = dp[b - 1, :i] + cost[:i, i]
            j = int(np.argmin(c))
            dp[b, i] = c[j]; prv[b, i] = j
    out = {}
    for B in Bs:
        edges = []
        i, b = N, B
        while b:
            j = int(prv[b, i]); edges.append(j); i, b = j, b - 1
        edges = edges[::-1]
        bits = dp[B, N] / max(hist.sum(), 1)
        out[B] = (edges, bits)
    return out

def apply_edges(g, edges):
    # edges are left indices into g+255. bin = searchsorted
    idx = g.astype(np.int32) + 255
    return np.searchsorted(edges, idx, side='right') - 1
