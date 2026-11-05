"""Shared Monte-Carlo sampling for llcomp_q7grid / q7calib / q7similar, cached on disk.

Per image: N random pixels (rng seeded by seed + file name, so a file always gives the same pixels whatever else is in
the folder), per plane Y, V, U (RCT V = R-G, U = B-G, Y = G + ((U+V) >> 2)): a = left, b = top, c = top-left
(x = 0: a = c = b; y = 0: b = c = 0), raw c-a, c-b (int16), the MED residual e wrapped (Y mod 256, U/V mod 512) and the residuals of the left and top
neighbours (E1 = e at x-1, E2 = e at y-1; 0 outside the image). Caches written before E1/E2 existed are rebuilt.
Cache: one .npz per image in --cache (default ~/.cache/clic-q7, on Windows C:\\Users\\<you>\\.cache\\clic-q7), named
<file>_<N>_<seed>.npz, reused while the image size and mtime match. --no-cache reads the images every time.
"""
import os, zlib
import numpy as np
from PIL import Image

DEFAULT_CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'clic-q7')

def add_args(ap):
    ap.add_argument('--cache', default=DEFAULT_CACHE, help='sample cache folder (default %(default)s)')
    ap.add_argument('--no-cache', action='store_true', help='do not read or write the cache')

def _med(P):
    """full-image MED residual (unwrapped) and c-a, c-b with the borders: x = 0: a = c = b; y = 0: b = c = 0"""
    b = np.zeros_like(P); b[1:] = P[:-1]
    a = np.empty_like(P); a[:, 1:] = P[:, :-1]; a[:, 0] = b[:, 0]
    c = np.zeros_like(P); c[1:, 1:] = P[:-1, :-1]; c[:, 0] = b[:, 0]
    return P - np.clip(a + b - c, np.minimum(a, b), np.maximum(a, b)), c - a, c - b

def sample_raw(path, n, seed):
    rng = np.random.default_rng([seed, zlib.crc32(os.path.basename(path).encode())])
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.int32)
    H, W = a.shape[:2]; ys = rng.integers(0, H, n); xs = rng.integers(0, W, n)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); out = []
    for P, n9 in ((Y, False), (V, True), (U, True)):
        E, GA, GB = _med(P)
        E = ((E + 256) & 511) - 256 if n9 else ((E + 128) & 255) - 128
        e = E[ys, xs]
        ea = np.where(xs > 0, E[ys, np.maximum(xs - 1, 0)], 0)          # residual of the left neighbour (0 at x = 0)
        eb = np.where(ys > 0, E[np.maximum(ys - 1, 0), xs], 0)          # residual of the top neighbour (0 at y = 0)
        out.append((int(n9), GA[ys, xs].astype(np.int16), GB[ys, xs].astype(np.int16), e.astype(np.int16),
                    ea.astype(np.int16), eb.astype(np.int16)))
    return out

def load(path, n, seed, cache=DEFAULT_CACHE, use_cache=True, e12=False):
    """[(t, c-a, c-b, e)] for Y, V, U (t = 0 for Y, 1 for U/V); e12=True: [(t, c-a, c-b, e, e_left, e_top)]"""
    r = _load(path, n, seed, cache, use_cache)
    return r if e12 else [x[:4] for x in r]

def _load(path, n, seed, cache, use_cache):
    if not use_cache: return sample_raw(path, n, seed)
    st = os.stat(path); cf = os.path.join(cache, f'{os.path.basename(path)}_{n}_{seed}.npz')
    if os.path.exists(cf):
        try:
            z = np.load(cf)
            if int(z['size']) == st.st_size and int(z['mtime']) == int(st.st_mtime) and 'ea0' in z.files:
                return [(int(z['t'][i]), z[f'ga{i}'], z[f'gb{i}'], z[f'e{i}'], z[f'ea{i}'], z[f'eb{i}']) for i in range(3)]
        except Exception: pass
    out = sample_raw(path, n, seed); os.makedirs(cache, exist_ok=True)
    d = {'size': st.st_size, 'mtime': int(st.st_mtime), 't': np.array([o[0] for o in out])}
    for i, (_, ga, gb, e, ea, eb) in enumerate(out): d[f'ga{i}'] = ga; d[f'gb{i}'] = gb; d[f'e{i}'] = e; d[f'ea{i}'] = ea; d[f'eb{i}'] = eb
    tmp = cf + '.tmp.npz'; np.savez_compressed(tmp, **d); os.replace(tmp, cf)
    return out
