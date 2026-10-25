"""The app's About-tab clip: one finished recording, every step side by side, as an animated WebP.

    python -m ms2.app.make_explainer <finished rec dir> [--out ms2/app/static/explainer.webp]

Each frame is one time point drawn five ways, next to each other, with the viewer's own
renderer: the raw cell channel, the segmentation, the tracks (a colour per cell, kept over
time), the MS2 spots, and the cells coloured by activity family.  A middle z-slice, cropped
at full resolution around where the MS2 spots are (single-pixel spots vanish when a frame is
scaled down).  Rendered once and committed; rerun it when the pipeline's look changes.
"""
import argparse
import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ms2.viewer.app import open_results
from ms2.viewer.render import _LUT, composite

SIZE = 320           # each panel, in image pixels (not scaled)
GAP = 8
HEAD = 52            # caption above each panel
FOOT = 34            # family legend below the row
FRAMES = 36          # time points
MS = 250             # per frame
# silent → highly active: grey, blue, green, orange, red
FAMILY_RGB = ((150, 150, 150), (60, 120, 255), (60, 220, 90), (255, 170, 30), (255, 50, 50))
BG = (17, 17, 17)


def font(px):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', px)
    except OSError:
        return ImageFont.load_default()


def by_track(v, t, mask):
    """The mask relabelled with track ids, so a cell's colour stays the same over time."""
    lut = np.zeros(int(mask.max()) + 1, dtype=np.int32)
    for label, tid in v.tracks.by_frame[t].items():
        if label < len(lut):
            lut[label] = tid + 1          # 0 stays background
    return lut[mask]


def crop_box(v):
    """A SIZE square (y0, x0) centred on the detected MS2 spots, else on the frame."""
    spots = [(p[1], p[0]) for ps in (v.peaks.by_t.values() if v.peaks else ()) for p in ps]
    cy, cx = np.median(spots, axis=0) if spots else (v.height / 2, v.width / 2)
    y0 = int(np.clip(cy - SIZE / 2, 0, max(0, v.height - SIZE)))
    x0 = int(np.clip(cx - SIZE / 2, 0, max(0, v.width - SIZE)))
    return y0, x0


def label_for(rgb):
    """The mask label composite draws in the palette colour closest to rgb."""
    i = int(np.argmin(((_LUT[1:].astype(int) - rgb) ** 2).sum(axis=1)))
    return i, tuple(int(c) for c in _LUT[i + 1])    # composite colours label L with _LUT[L % 255 + 1]


def row(panels, captions, legend, t, n_t):
    """The panels side by side, a caption over each, the frame and the family legend below."""
    w = len(panels) * SIZE + (len(panels) + 1) * GAP
    out = Image.new('RGB', (w, HEAD + SIZE + FOOT), BG)
    d = ImageDraw.Draw(out)
    for i, (png, (title, line)) in enumerate(zip(panels, captions)):
        x = GAP + i * (SIZE + GAP)
        out.paste(Image.open(io.BytesIO(png)).convert('RGB'), (x, HEAD))
        d.text((x, 8), title, font=font(17), fill=(102, 204, 255))
        d.text((x, 30), line, font=font(12), fill=(200, 200, 200))
    y = HEAD + SIZE + 10
    d.text((GAP, y), f'frame {t + 1} of {n_t}', font=font(12), fill=(140, 140, 140))
    f = font(12)
    x = w - GAP - sum(22 + d.textlength(name, font=f) for name, _ in legend) + 8   # right-aligned
    for name, rgb in legend:
        d.rectangle((x, y + 2, x + 10, y + 12), fill=rgb)
        d.text((x + 14, y), name, font=f, fill=(200, 200, 200))
        x += 22 + d.textlength(name, font=f)
    return out


def render(rec_dir):
    v = open_results(rec_dir)
    z = v.n_z // 2
    y0, x0 = crop_box(v)
    cut = np.s_[y0:y0 + SIZE, x0:x0 + SIZE]
    levels = v.families.summary()['levels'] if v.families is not None else []
    picked = {f['level']: label_for(FAMILY_RGB[min(f['level'], len(FAMILY_RGB) - 1)]) for f in levels}
    fam_label = {level: label for level, (label, _) in picked.items()}
    legend = [(f['name'], picked[f['level']][1]) for f in levels]

    captions = [('1 · Recording', 'cell nuclei over time'),
                ('2 · Segmentation', 'every nucleus found in 3D'),
                ('3 · Tracking', 'each cell keeps its colour'),
                ('4 · Gene expression', 'MS2 spots (yellow) in their cell'),
                ('5 · Activity', 'cells grouped by how they transcribe')]

    frames = []
    for t in np.linspace(0, v.n_t - 1, FRAMES).astype(int):
        lo, hi = v.window(t, 40.0, 99.7)   # most pixels are background: keep its noise black
        img = v.image[t, z][cut]
        mask = v.mask_slice(t, z)[cut]
        ms2 = v.ms2.get(t) if v.ms2 is not None else None
        peaks = [(x - x0, y - y0, label, score)
                 for x, y, label, score in (v.peaks.get(t) if v.peaks is not None else [])
                 if 0 <= x - x0 < SIZE and 0 <= y - y0 < SIZE]
        # every family at once, one colour each, filled so it reads at a glance
        family = np.zeros_like(mask)
        for level, label in fam_label.items():
            cells = [lab for lab in (v.tracks.state_at(tid, t)[0] for tid in v.families.tids(level))
                     if lab > 0]
            family[np.isin(mask, cells)] = label
        panels = [composite(img, None, lo, hi),
                  composite(img, mask, lo, hi, mode='outline'),
                  composite(img, by_track(v, t, mask), lo, hi, alpha=0.45, mode='fill'),
                  composite(img, mask, lo, hi, mode='outline',
                            ms2=None if ms2 is None else ms2[cut], peaks=peaks),
                  composite(img, family, lo, hi, alpha=0.6, mode='fill')]
        frames.append(row(panels, captions, legend, t, v.n_t))
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('rec_dir', help='a finished recording folder (all six steps done)')
    p.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'static', 'explainer.webp'))
    args = p.parse_args()
    frames = render(args.rec_dir)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=MS, loop=0,
                   quality=60, method=6)
    print(f'{args.out}: {len(frames)} frames, {os.path.getsize(args.out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
