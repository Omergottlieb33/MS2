"""CZI -> the per-channel tifs the pipeline reads, replacing the manual ImageJ steps.

    python -m src.preprocess <rec>.czi --out <dir>

writes, into <dir>:
    C1-<rec>.tif             MS2 channel (czi channel 0), T x Z x Y x X, uint8
    C2-<rec>.tif             cell channel (czi channel 1)
    C1-<rec>_bg_removed.tif  C1 after ImageJ's Process > Subtract Background

The background subtraction is a port of ImageJ's BackgroundSubtracter, sliding-paraboloid
path, with ImageJ's defaults for everything else (dark background, smoothing and corner
correction on), run on every (t, z) slice.  It reproduces ImageJ's 8-bit output pixel for
pixel; see tests/test_preprocess.py.  Smoothing matters: it shifts the background down by
twice the mean lift of the 3x3 maximum, which is why empty pixels come out as 1, not 0.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import czifile
import numpy as np
import tifffile
from numba import njit

MS2_CHANNEL = 0
CELL_CHANNEL = 1
BG_RADIUS = 5

F = np.float32
X_DIRECTION, Y_DIRECTION, DIAGONAL_1A, DIAGONAL_1B, DIAGONAL_2A, DIAGONAL_2B = range(6)


@njit(cache=True, nogil=True)
def _line_slide_parabola(pixels, start, inc, length, coeff2, cache, next_point, edges, want_edges):
    """BackgroundSubtracter.lineSlideParabola, without the NaN handling (input is integer)."""
    min_value = F(3.4028235e38)
    lastpoint = 0
    vp1 = F(0.0)
    vp2 = F(0.0)
    curv = F(1.999) * coeff2
    p = start
    for i in range(length):
        v = pixels[p]
        cache[i] = v
        if v < min_value:
            min_value = v
        if i >= 2 and (vp1 + vp1 - vp2 - v) < curv:
            next_point[lastpoint] = i - 1
            lastpoint = i - 1
        vp2 = vp1
        vp1 = v
        p += inc
    next_point[lastpoint] = length - 1
    next_point[length - 1] = 2147483647
    first_corner = length
    last_corner = 0
    i1 = 0
    while i1 < length - 1:
        v1 = cache[i1]
        min_slope = F(3.4028235e38)
        i2 = 0
        search_to = length
        recalc = 0
        j = next_point[i1]
        while j < search_to:
            slope = F(F(cache[j] - v1) / F(j - i1)) + F(coeff2 * F(j - i1))
            if slope < min_slope:
                min_slope = slope
                i2 = j
                recalc = -3
            if recalc == 0:
                b = np.float64(F(F(0.5) * min_slope) / coeff2)
                max_search = i1 + int(b + np.sqrt(b * b + np.float64(F(v1 - min_value) / coeff2)) + 1)
                if max_search < search_to and max_search > 0:
                    search_to = max_search
            j = next_point[j]
            recalc += 1
        if i2 <= i1:
            i2 = length - 1
        if first_corner >= length and i1 > 0:
            first_corner = i1
        if i2 < length - 1:
            last_corner = i2
        p = start + (i1 + 1) * inc
        for jj in range(i1 + 1, i2):
            d = F(jj - i1)
            pixels[p] = v1 + d * (min_slope - d * coeff2)
            p += inc
        i1 = i2
    if want_edges:
        if first_corner > last_corner:
            edges[0] = np.nan
            edges[1] = np.nan
        else:
            if 4 * first_corner >= length:
                first_corner = 0
            if 4 * (length - 1 - last_corner) >= length:
                last_corner = length - 1
            v1 = cache[first_corner]
            v2 = cache[last_corner]
            if last_corner - first_corner > length // 4 + 1:
                slope = F(v2 - v1) / F(last_corner - first_corner)
            else:
                slope = F(np.nan)
            value0 = v1 - slope * F(first_corner)
            coeff6 = F(0.0)
            mid = F(0.5) * F(last_corner + first_corner)
            for i in range((length + 2) // 3, (2 * length) // 3 + 1):
                dx = F(F(i) - mid) * F(2.0) / F(last_corner - first_corner)
                poly6 = dx * dx * dx * dx * dx * dx - F(1.0)
                if cache[i] < value0 + slope * F(i) + coeff6 * poly6:
                    coeff6 = -(value0 + slope * F(i) - cache[i]) / poly6
            dx = F(F(first_corner) - mid) * F(2.0) / F(last_corner - first_corner)
            edges[0] = (value0 + coeff6 * (dx * dx * dx * dx * dx * dx - F(1.0))
                        + coeff2 * F(first_corner) * F(first_corner))
            dx = F(F(last_corner) - mid) * F(2.0) / F(last_corner - first_corner)
            e = F(length - 1 - last_corner)
            edges[1] = (value0 + F(length - 1) * slope
                        + coeff6 * (dx * dx * dx * dx * dx * dx - F(1.0)) + coeff2 * e * e)


@njit(cache=True, nogil=True)
def _filter1d(pixels, width, height, direction, coeff2, cache, next_point, edges):
    start_line, n_lines, line_inc, point_inc, length = 0, 0, 0, 0, 0
    if direction == X_DIRECTION:
        n_lines, line_inc, point_inc, length = height, width, 1, width
    elif direction == Y_DIRECTION:
        n_lines, line_inc, point_inc, length = width, 1, width, height
    elif direction == DIAGONAL_1A:
        n_lines, line_inc, point_inc = width - 2, 1, width + 1
    elif direction == DIAGONAL_1B:
        start_line, n_lines, line_inc, point_inc = 1, height - 2, width, width + 1
    elif direction == DIAGONAL_2A:
        start_line, n_lines, line_inc, point_inc = 2, width, 1, width - 1
    else:
        n_lines, line_inc, point_inc = height - 2, width, width - 1
    for i in range(start_line, n_lines):
        start = i * line_inc
        if direction == DIAGONAL_2B:
            start += width - 1
        if direction == DIAGONAL_1A:
            length = min(height, width - i)
        elif direction == DIAGONAL_1B or direction == DIAGONAL_2B:
            length = min(width, height - i)
        elif direction == DIAGONAL_2A:
            length = min(height, i + 1)
        _line_slide_parabola(pixels, start, point_inc, length, coeff2, cache, next_point, edges, False)


@njit(cache=True, nogil=True)
def _filter3x3(pixels, width, height, is_max):
    """BackgroundSubtracter.filter3x3: rows then columns; returns the mean shift of the maximum."""
    shift_sum = 0.0
    n = 0.0
    for pass_ in range(2):
        if pass_ == 0:
            n_lines, length, line_inc, inc = height, width, width, 1
        else:
            n_lines, length, line_inc, inc = width, height, 1, width
        for li in range(n_lines):
            p = li * line_inc
            v3 = pixels[p]
            v2 = v3
            for i in range(length):
                v1 = v2
                v2 = v3
                if i < length - 1:
                    v3 = pixels[p + inc]
                if is_max:
                    mx = v1 if v1 > v3 else v3
                    if v2 > mx:
                        mx = v2
                    shift_sum += np.float64(mx - v2)
                    n += 1.0
                    pixels[p] = mx
                else:
                    pixels[p] = (v1 + v2 + v3) * F(0.333333333)
                p += inc
    return shift_sum / n if n > 0 else 0.0


def _correct_corners(pix, w, h, coeff2, cache, nxt):
    edges = np.zeros(2, F)
    corners = np.zeros(4, F)
    counts = np.zeros(4, np.int64)

    def add(i, v):
        if not np.isnan(v):
            corners[i] += v
            counts[i] += 1

    def edge_estimates(start, inc, length, c2):
        _line_slide_parabola(pix, start, inc, length, F(c2), cache, nxt, edges, True)
        return edges[0], edges[1]

    a, b = edge_estimates(0, 1, w, coeff2); add(0, a); add(1, b)
    a, b = edge_estimates((h - 1) * w, 1, w, coeff2); add(2, a); add(3, b)
    a, b = edge_estimates(0, w, h, coeff2); add(0, a); add(2, b)
    a, b = edge_estimates(w - 1, w, h, coeff2); add(1, a); add(3, b)
    diag, c2diag = min(w, h), F(2) * coeff2
    add(0, edge_estimates(0, 1 + w, diag, c2diag)[0])
    add(1, edge_estimates(w - 1, w - 1, diag, c2diag)[0])
    add(2, edge_estimates((h - 1) * w, 1 - w, diag, c2diag)[0])
    add(3, edge_estimates(w * h - 1, -1 - w, diag, c2diag)[0])
    avg = corners / counts.astype(F)
    for idx, k in ((0, 0), (w - 1, 1), ((h - 1) * w, 2), (w * h - 1, 3)):
        if pix[idx] > avg[k] or np.isnan(pix[idx]):
            pix[idx] = avg[k]


def paraboloid_background(img, radius=BG_RADIUS):
    """ImageJ's sliding-paraboloid background of a 2D image, as float32."""
    h, w = img.shape
    pix = img.astype(F).ravel()
    cache = np.zeros(max(w, h), F)
    nxt = np.zeros(max(w, h), np.int64)
    edges = np.zeros(2, F)
    coeff2 = F(F(0.5) / F(radius))
    coeff2diag = F(F(1.0) / F(radius))
    shift_by = F(_filter3x3(pix, w, h, True))   # 3x3 maximum against dust
    _filter3x3(pix, w, h, False)                # 3x3 mean against noise
    _correct_corners(pix, w, h, coeff2, cache, nxt)
    for d, c in ((X_DIRECTION, coeff2), (Y_DIRECTION, coeff2), (X_DIRECTION, coeff2),
                 (DIAGONAL_1A, coeff2diag), (DIAGONAL_1B, coeff2diag),
                 (DIAGONAL_2A, coeff2diag), (DIAGONAL_2B, coeff2diag),
                 (DIAGONAL_1A, coeff2diag), (DIAGONAL_1B, coeff2diag)):
        _filter1d(pix, w, h, d, c, cache, nxt, edges)
    pix -= F(2) * shift_by                      # undo the lift of the 3x3 maximum
    return pix.reshape(h, w)


def subtract_background(img, radius=BG_RADIUS):
    """One 8-bit slice, as ImageJ writes it: raw - background + 0.5, clamped, truncated."""
    v = img.astype(F) - paraboloid_background(img, radius) + F(0.5)
    return np.clip(v, 0, 255).astype(np.uint8)


def subtract_background_stack(stack, radius=BG_RADIUS, workers=None):
    """Every 2D slice of an (..., Y, X) uint8 stack, in parallel threads (the kernels drop the GIL)."""
    if stack.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit stack, got {stack.dtype}")
    flat = stack.reshape(-1, *stack.shape[-2:])
    out = np.empty_like(flat)

    def one(i):
        out[i] = subtract_background(flat[i], radius)

    with ThreadPoolExecutor(workers or os.cpu_count()) as pool:
        list(pool.map(one, range(len(flat))))
    return out.reshape(stack.shape)


def load_czi(path):
    """(T, C, Z, Y, X); czifile adds singleton axes (scenes, samples) that are squeezed away."""
    with czifile.CziFile(path) as czi:
        # older czifile has .axes on the file; newer (2025+) only on the scene
        axes = czi.axes if hasattr(czi, 'axes') else czi.scenes[0].axes
        data = czi.asarray()
    keep = [i for i, a in enumerate(axes) if a in 'TCZYX']
    drop = tuple(i for i in range(data.ndim) if i not in keep)
    data = data.squeeze(axis=drop)
    order = [a for a in axes if a in 'TCZYX']
    if order != list('TCZYX'):
        raise ValueError(f"unexpected czi axes {axes}; expected T, C, Z, Y, X")
    return data


def write_imagej(path, stack):
    tifffile.imwrite(path, stack, imagej=True, metadata={'axes': 'TZYX'})


def preprocess(czi_path, out_dir, radius=BG_RADIUS, ms2_channel=MS2_CHANNEL,
               cell_channel=CELL_CHANNEL, workers=None):
    rec = os.path.splitext(os.path.basename(czi_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    data = load_czi(czi_path)
    ms2, cells = data[:, ms2_channel], data[:, cell_channel]
    paths = {
        'ms2': os.path.join(out_dir, f'C1-{rec}.tif'),
        'cells': os.path.join(out_dir, f'C2-{rec}.tif'),
        'ms2_bg_removed': os.path.join(out_dir, f'C1-{rec}_bg_removed.tif'),
    }
    write_imagej(paths['ms2'], ms2)
    write_imagej(paths['cells'], cells)
    write_imagej(paths['ms2_bg_removed'], subtract_background_stack(ms2, radius, workers))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('czi')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--radius', type=float, default=BG_RADIUS)
    ap.add_argument('--ms2-channel', type=int, default=MS2_CHANNEL)
    ap.add_argument('--cell-channel', type=int, default=CELL_CHANNEL)
    ap.add_argument('--workers', type=int)
    args = ap.parse_args()
    paths = preprocess(args.czi, args.out, args.radius, args.ms2_channel, args.cell_channel, args.workers)
    for p in paths.values():
        print(p)


if __name__ == '__main__':
    main()
