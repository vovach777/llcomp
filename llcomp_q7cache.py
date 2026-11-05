"""Shared Monte-Carlo sampling for llcomp_q7grid / q7calib / q7similar, cached on disk.

Per image: N random pixels (rng seeded by seed + file name, so a file always gives the same pixels whatever else is in
the folder), per plane Y, V, U (RCT V = R-G, U = B-G, Y = G + ((U+V) >> 2)): a = left, b = top, c = top-left
(x = 0: a = c = b; y = 0: b = c = 0), raw c-a, c-b (int16) and the MED residual e wrapped (Y mod 256, U/V mod 512).
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

def sample_raw(path, n, seed):
    rng = np.random.default_rng([seed, zlib.crc32(os.path.basename(path).encode())])
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.int32)
    H, W = a.shape[:2]; ys = rng.integers(0, H, n); xs = rng.integers(0, W, n)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]; V = R - G; U = B - G; Y = G + ((U + V) >> 2); out = []
    for P, n9 in ((Y, False), (V, True), (U, True)):
        p = P[ys, xs]
        b = np.where(ys > 0, P[np.maximum(ys - 1, 0), xs], 0)
        a_ = np.where(xs > 0, P[ys, np.maximum(xs - 1, 0)], b)
        c = np.where((ys > 0) & (xs > 0), P[np.maximum(ys - 1, 0), np.maximum(xs - 1, 0)], b); c = np.where(xs == 0, b, c)
        e = p - np.clip(a_ + b - c, np.minimum(a_, b), np.maximum(a_, b))
        e = ((e + 256) & 511) - 256 if n9 else ((e + 128) & 255) - 128
        out.append((int(n9), (c - a_).astype(np.int16), (c - b).astype(np.int16), e.astype(np.int16)))
    return out

def load(path, n, seed, cache=DEFAULT_CACHE, use_cache=True):
    """[(t, c-a, c-b, e)] for Y, V, U (t = 0 for Y, 1 for U/V)"""
    if not use_cache: return sample_raw(path, n, seed)
    st = os.stat(path); cf = os.path.join(cache, f'{os.path.basename(path)}_{n}_{seed}.npz')
    if os.path.exists(cf):
        try:
            z = np.load(cf)
            if int(z['size']) == st.st_size and int(z['mtime']) == int(st.st_mtime):
                return [(int(z['t'][i]), z[f'ga{i}'], z[f'gb{i}'], z[f'e{i}']) for i in range(3)]
        except Exception: pass
    out = sample_raw(path, n, seed); os.makedirs(cache, exist_ok=True)
    d = {'size': st.st_size, 'mtime': int(st.st_mtime), 't': np.array([o[0] for o in out])}
    for i, (_, ga, gb, e) in enumerate(out): d[f'ga{i}'] = ga; d[f'gb{i}'] = gb; d[f'e{i}'] = e
    tmp = cf + '.tmp.npz'; np.savez_compressed(tmp, **d); os.replace(tmp, cf)
    return out
