"""Is one recording an outlier of the comparison only because of its photometry?

src/cell_activity_compare.py measures every recording against one absolute noise floor on the
raw ellipse sums.  If one recording was imaged brighter, it looks more active for that reason
alone.  This reruns the comparison under normalisations that remove more and more of the
intensity difference, and scores the tested recording as an outlier under each:

    baseline             the raw matrices, one floor -- the comparison as it was
    noise_normalized     each matrix scaled by pooled / own median background sigma, the
                         `noise` column of the per-cell *_final.csv files (robust sigma of the
                         annulus around every accepted emitter).  Matches the recordings'
                         background photometry and nothing else.
    quantile_normalized  each recording's nonzero values mapped onto one reference distribution,
                         the mean of the recordings' quantile functions.  Treats EVERY amplitude
                         difference as technical, so it is the upper bound of what a
                         normalisation can remove; detection frequency is untouched.
    floor_<f>            the raw matrices with the tested recording's floor raised to f and
                         the others left where they are.

Every variant is a full compare run in out_dir/<variant>.  outlier_summary.csv scores the
tested recording per variant and metric:

    dixon_q              gap to the nearest other recording over the range of all of them, 0
                         when it is not the extreme one.  With three recordings the critical
                         values are 0.941 (p=0.10), 0.970 (0.05), 0.994 (0.01); three embryos
                         make any embryo-level test weak, so read it with the ratio.
    ratio                its value over the mean of the others
    p_cells              cells, not embryos (see stats_readme.txt): Mann-Whitney of its active
                         cells against the others' pooled, Fisher's exact for frac_active

    python -m src.cell_activity_normalize --config recordings.json --target New-17-ST12-V \\
        --out-dir .../normalization_outlier
"""
import argparse
import glob
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, rankdata

from src.cell_activity import K_LEVELS, NOISE_FLOOR, save_figure
from src.cell_activity_compare import (INK, MUTED, compare_recordings, recording_colors,
                                       style_axes)

FLOORS = (25, 30, 40, 50, 60, 80, 100)
METRICS = ('frac_active', 'median_rate', 'median_peak', 'median_duty', 'median_onset',
           'mean_frac_firing')
# The per-cell column each metric summarises, for the cell-level test
CELL_METRIC = {'median_rate': 'rate', 'median_peak': 'peak', 'median_duty': 'duty',
               'median_onset': 'onset'}
DIXON_CRITICAL = {0.10: 0.941, 0.05: 0.970, 0.01: 0.994}  # n = 3


def median_noise(csv_path):
    """Median background sigma over the accepted emitters of the matrix's per-cell files."""
    frames = [pd.read_csv(f) for f in
              glob.glob(os.path.join(os.path.dirname(csv_path), 'cell_*_final.csv'))]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    return float(df.loc[df['ellipse_sum'] > 0, 'noise'].median())


def write_matrix(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    return path


def read_matrix(csv_path):
    return pd.read_csv(csv_path).set_index('timepoint')


def noise_normalized(recordings, matrix_dir):
    noise = {name: median_noise(rec['csv']) for name, rec in recordings.items()}
    pooled = float(np.median(list(noise.values())))
    out = {}
    for name, rec in recordings.items():
        scaled = read_matrix(rec['csv']) * (pooled / noise[name])
        out[name] = dict(rec, csv=write_matrix(scaled, os.path.join(matrix_dir, f'{name}.csv')))
    print('median background sigma: ' + ', '.join(f'{n} {v:.2f}' for n, v in noise.items())
          + f' -> scaled to {pooled:.2f}')
    return out, {f'scale_{n}': pooled / v for n, v in noise.items()}


def quantile_normalized(recordings, matrix_dir, n_quantiles=1001):
    matrices = {name: read_matrix(rec['csv']) for name, rec in recordings.items()}
    q = np.linspace(0, 1, n_quantiles)
    nonzero = {name: m.to_numpy()[m.to_numpy() > 0] for name, m in matrices.items()}
    reference = np.mean([np.quantile(v, q) for v in nonzero.values()], axis=0)
    out = {}
    for name, m in matrices.items():
        values = m.to_numpy().copy()
        mask = values > 0
        # the value's own quantile, ties averaged, read off the reference quantile function
        values[mask] = np.interp((rankdata(values[mask]) - 0.5) / mask.sum(), q, reference)
        mapped = pd.DataFrame(values, index=m.index, columns=m.columns)
        out[name] = dict(recordings[name],
                         csv=write_matrix(mapped, os.path.join(matrix_dir, f'{name}.csv')))
    return out, {}


def raised_floor(recordings, target, floor):
    return {name: dict(rec, noise_floor=floor) if name == target else rec
            for name, rec in recordings.items()}, {}


def dixon_q(values, target):
    others = values.drop(target)
    spread = values.max() - values.min()
    if len(values) < 3 or spread == 0 or not (values[target] in (values.max(), values.min())):
        return 0.0
    return float(np.min(np.abs(others - values[target])) / spread)


def outlier_rows(variant, out_dir, target):
    metrics = pd.read_csv(os.path.join(out_dir, 'metrics.csv')).set_index('recording')
    cells = pd.read_csv(os.path.join(out_dir, 'cells.csv'))
    is_target = cells['recording'] == target
    rows = []
    for metric in METRICS:
        values = metrics[metric]
        row = {'variant': variant, 'metric': metric, 'target': values[target],
               'others_mean': values.drop(target).mean(),
               'ratio': values[target] / values.drop(target).mean(),
               'dixon_q': dixon_q(values, target)}
        row.update({f'{name}': v for name, v in values.items()})
        if metric == 'frac_active':
            table = [[cells.loc[is_target, 'active'].sum(), (~cells.loc[is_target, 'active']).sum()],
                     [cells.loc[~is_target, 'active'].sum(), (~cells.loc[~is_target, 'active']).sum()]]
            row['p_cells'] = fisher_exact(table)[1]
        elif metric in CELL_METRIC:
            a = cells.loc[is_target & cells['active'], CELL_METRIC[metric]].dropna()
            b = cells.loc[~is_target & cells['active'], CELL_METRIC[metric]].dropna()
            row['p_cells'] = mannwhitneyu(a, b)[1] if len(a) and len(b) else np.nan
        rows.append(row)
    return rows


def plot_summary(summary, names, target, path_stem):
    """One panel per metric: each recording's value across the variants, the target in its
    recording colour and bold."""
    variants = list(dict.fromkeys(summary['variant']))
    colors = recording_colors(names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(variants))
    for ax, metric in zip(axes.ravel(), METRICS):
        s = summary[summary['metric'] == metric].set_index('variant').loc[variants]
        for name in names:
            ax.plot(x, s[name], color=colors[name], lw=2.5 if name == target else 1.5,
                    marker='o', markersize=5, label=name)
        for xi, q in zip(x, s['dixon_q']):
            if q >= DIXON_CRITICAL[0.10]:
                ax.text(xi, ax.get_ylim()[1], '*', ha='center', va='top', color=INK, fontsize=12)
        ax.set_xticks(x, variants, rotation=40, ha='right', fontsize=8)
        ax.set_title(metric, fontsize=10)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{target} against the others under each normalisation '
                 f'(* Dixon Q >= {DIXON_CRITICAL[0.10]}, p < 0.10 with n = 3)', color=MUTED)
    fig.tight_layout()
    save_figure(fig, path_stem)


def run(recordings, target, out_dir, k=K_LEVELS, floors=FLOORS):
    variants = {'baseline': lambda: (recordings, {}),
                'noise_normalized': lambda: noise_normalized(
                    recordings, os.path.join(out_dir, 'matrices', 'noise_normalized')),
                'quantile_normalized': lambda: quantile_normalized(
                    recordings, os.path.join(out_dir, 'matrices', 'quantile_normalized'))}
    base_floor = recordings[target].get('noise_floor', NOISE_FLOOR)
    for f in floors:
        variants[f'floor_{f:g}'] = lambda f=f: raised_floor(recordings, target, f)

    rows, notes, first_cache = [], {}, None
    for variant, build in variants.items():
        print(f'\n=== {variant}')
        recs, notes[variant] = build()
        variant_dir = os.path.join(out_dir, variant)
        cache = os.path.join(variant_dir, 'cache')
        # cell positions depend on the masks only, so every variant after the first reuses them
        if first_cache:
            os.makedirs(cache, exist_ok=True)
            for pkl in glob.glob(os.path.join(first_cache, '*_cell_properties.pkl')):
                shutil.copy(pkl, cache)
        first_cache = first_cache or cache
        with open(os.path.join(out_dir, f'recordings_{variant}.json'), 'w') as fh:
            json.dump(recs, fh, indent=2)
        compare_recordings(recs, variant_dir, k=k)
        rows += outlier_rows(variant, variant_dir, target)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'outlier_summary.csv'), index=False)
    with open(os.path.join(out_dir, 'notes.json'), 'w') as fh:
        json.dump({'target': target, 'target_base_floor': base_floor, 'k': k,
                   'dixon_critical_n3': DIXON_CRITICAL, 'variant_notes': notes}, fh, indent=2)
    plot_summary(summary, list(recordings), target, os.path.join(out_dir, 'outlier_summary'))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='json of {name: recording dict}')
    p.add_argument('--target', required=True, help='the recording tested as an outlier')
    p.add_argument('--out-dir', required=True, help='one compare run per variant goes here')
    p.add_argument('--k', type=int, default=K_LEVELS,
                   help='ladder levels, as in cell_activity_compare (default %(default)s)')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        recordings = json.load(f)
    summary = run(recordings, args.target, args.out_dir, args.k)
    pd.set_option('display.width', 200)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
