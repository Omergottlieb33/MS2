"""Compositing of a single (t, z) slice into a PNG for the viewer."""
import io

import numpy as np
from PIL import Image

from src.utils.plot_utils import blend, generate_random_colors

# Colours are looked up by label value, not by position in the frame's label list,
# so a cell keeps its colour across timepoints.  Index 0 is reserved for background,
# hence the 1..255 range.
_LUT = np.zeros((256, 3), dtype=np.uint8)
_LUT[1:] = np.array(generate_random_colors(255), dtype=np.uint8)

# A selected track keeps ONE colour for its whole life.  Its label changes from frame to
# frame by design, so colouring the highlight from the label LUT would make the cell you
# are following change colour as you scrub -- exactly the wrong signal.  generate_random_colors
# never emits a channel below 50, so pure red cannot collide with a neighbour.
HIGHLIGHT = np.array([255, 0, 0], dtype=np.uint8)


def _dilate(edge):
    """Grow a boolean mask by one pixel, so a selected cell's border stays visible
    when zoomed out."""
    out = edge.copy()
    out[:-1, :] |= edge[1:, :]
    out[1:, :] |= edge[:-1, :]
    out[:, :-1] |= edge[:, 1:]
    out[:, 1:] |= edge[:, :-1]
    return out


def _grow(a):
    """Greyscale dilation by one pixel: each pixel takes the max of its 4 neighbours."""
    out = a.copy()
    out[:-1, :] = np.maximum(out[:-1, :], a[1:, :])
    out[1:, :] = np.maximum(out[1:, :], a[:-1, :])
    out[:, :-1] = np.maximum(out[:, :-1], a[:, 1:])
    out[:, 1:] = np.maximum(out[:, 1:], a[:, :-1])
    return out


def _outlines(mask):
    """Boundary pixels of labelled regions, vectorised over the whole slice."""
    diff_y = mask[:-1, :] != mask[1:, :]
    diff_x = mask[:, :-1] != mask[:, 1:]
    edge = np.zeros(mask.shape, dtype=bool)
    edge[:-1, :] |= diff_y
    edge[1:, :] |= diff_y
    edge[:, :-1] |= diff_x
    edge[:, 1:] |= diff_x
    return edge & (mask > 0)


def intensity_window(volume, lo_pct, hi_pct):
    """Intensity window for a whole (Z, Y, X) timepoint, so brightness does not
    flicker while scrolling z.  Subsampled: percentiles are stable under it and the
    full volume is 10M pixels."""
    lo, hi = np.percentile(volume.ravel()[::16], [lo_pct, hi_pct])
    lo, hi = float(lo), float(hi)
    return lo, hi if hi > lo else lo + 1.0


def _draw_all(rgb, mask_2d, colors, alpha, mode):
    if mode == 'outline':
        sel = _outlines(mask_2d)
        rgb[sel] = colors[sel]            # outlines stay opaque so thin borders read clearly
    else:
        sel = mask_2d > 0
        rgb[sel] = blend(rgb[sel], colors[sel], alpha).astype(np.uint8)


def _draw_highlight(rgb, mask_2d, colors, highlight):
    """One cell picked out of the crowd: everything else drops to a faint outline so
    the neighbours that explain a bad match stay visible."""
    others = np.where((mask_2d > 0) & (mask_2d != highlight), mask_2d, 0)
    faint = _outlines(others)
    # Dimmed colour at fairly high alpha, not full colour at low alpha: cell interiors
    # are near-white, so a low-alpha blend washes out to nothing and the context is lost.
    rgb[faint] = blend(rgb[faint], colors[faint] * 0.7, 0.6).astype(np.uint8)

    if highlight <= 0:
        return          # track is in a gap or gone; nothing to pick out.  Must come before
                        # the compare below -- `mask_2d == 0` is the whole background.

    sel = mask_2d == highlight
    if not sel.any():
        return
    edge = _dilate(_outlines(np.where(sel, mask_2d, 0)))
    rgb[sel] = blend(rgb[sel], HIGHLIGHT, 0.45).astype(np.uint8)
    rgb[edge] = HIGHLIGHT


def _draw_ms2(rgb, ms2, thr):
    """MS2 signal as a yellow glow: red and green pushed up, blue pulled down.

    Pulling blue down is what makes this work.  Emitters sit inside nuclei, which are
    already near-white in the cell channel, and adding red+green to a white pixel changes
    nothing -- the spots that matter most would be invisible.  Removing blue turns white
    into yellow, so a spot reads on bright and dark background alike.

    Everything below `thr` is left untouched: the background-subtracted channel still has a
    floor across ~95% of the frame, and drawing that would paint the whole field yellow.
    """
    # Ramp to full yellow only ~8 above the threshold.  Detected emitters run about 21-68
    # against a default threshold of 18, so a gentler ramp would leave the typical spot at
    # a third brightness -- present in the pixels, invisible on screen.
    strength = np.clip((ms2.astype(np.float32) - thr) / 8.0, 0, 1)
    lit = strength > 0
    if not lit.any():
        return
    # An emitter is often a single pixel, which no one can see in a 1024x1024 view.  Grow
    # each spot by one pixel so it reads as a dot; this changes its apparent size, never
    # which cell it falls in.
    strength = _grow(strength)
    lit = strength > 0
    s = strength[lit][:, None]
    px = rgb[lit].astype(np.float32)
    px[:, :2] = np.clip(px[:, :2] + 255.0 * s, 0, 255)
    px[:, 2] *= (1.0 - s[:, 0])
    rgb[lit] = px.astype(np.uint8)


# Ring offsets enclosing exactly the 3x3 block around a peak: every pixel within a radius
# of 2 of the centre is one of those 9, so the circle sits just outside them and leaves the
# emitter itself unpainted.
_PEAK_YY, _PEAK_XX = np.ogrid[-3:4, -3:4]
_PEAK_D = np.sqrt(_PEAK_XX ** 2 + _PEAK_YY ** 2)
_PEAK_RING = np.where((_PEAK_D >= 2.0) & (_PEAK_D < 2.6))
_PEAK_DY, _PEAK_DX = _PEAK_RING[0] - 3, _PEAK_RING[1] - 3


def _draw_peaks(rgb, peaks):
    """Detected emitters as small yellow circles, each enclosing 9 pixels.

    An outline rather than a blob, because the thing worth seeing is *how many* emitters a
    nucleus has -- two spots a few pixels apart merge into one shape if drawn solid, which
    is the same failure as thresholding the raw channel.  Nothing is drawn inside the ring,
    so the signal at the peak stays visible.
    """
    h, w = rgb.shape[:2]
    for x, y, _label, _score in peaks:
        ys, xs = int(round(y)) + _PEAK_DY, int(round(x)) + _PEAK_DX
        keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        rgb[ys[keep], xs[keep]] = (255, 255, 0)


def composite(img_2d, mask_2d, lo, hi, alpha=0.4, mode='outline', highlight=None,
              ms2=None, ms2_thr=18.0, peaks=None):
    """Contrast-stretched greyscale slice with the mask drawn over it, as PNG bytes.

    highlight: a label id to pick out, or None to draw every cell equally.
    ms2:       2D MS2 map to lay over in yellow, or None.
    peaks:     detected emitters [(x, y, label, score)] to ring, or None.
    """
    scaled = (img_2d.astype(np.float32) - lo) * (255.0 / (hi - lo))
    rgb = np.repeat(np.clip(scaled, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)

    if mask_2d is not None and alpha > 0:
        colors = _LUT[(mask_2d % 255 + 1).astype(np.uint8)]
        if highlight is None:
            _draw_all(rgb, mask_2d, colors, alpha, mode)
        else:
            _draw_highlight(rgb, mask_2d, colors, highlight)

    if ms2 is not None:
        # last, on top of the masks.  An emitter is often a single pixel, so anything drawn
        # over it wins -- mask outlines were hiding roughly a fifth of the detected spots.
        _draw_ms2(rgb, ms2, ms2_thr)

    if peaks:
        _draw_peaks(rgb, peaks)

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG', compress_level=1)
    return buf.getvalue()
