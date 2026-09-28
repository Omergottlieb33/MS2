"""Why activity_levels picks the k it picks, and how much that choice can be trusted.

activity_levels (src/cell_activity.py) fits KMeans on log1p(score) for k = 2..N_LEVELS and keeps
the k with the best silhouette.  This rebuilds every fit a run makes -- one ladder per recording
on total, as src/cell_activity.py does, and the pooled ladder on rate, as
src/cell_activity_compare.py does -- and asks of each:

    margin          best silhouette minus the runner-up.  A few thousandths is a coin flip that
                    max() settles by taking the smaller k.
    seed spread     the silhouette over KMeans seeds; activity_levels uses seed 0 only.
    bootstrap       how often each k wins over resamples of the cells -- the choice's stability.
    gmm_bic_k       k=1..K_MAX by Gaussian-mixture BIC.  The silhouette cannot score k=1, so it
                    returns at least two levels even from one unimodal blob; BIC can say "one".
    frac_score_lt_1 share of scores below 1, where log1p is close to linear rather than a log.
                    rate (per tracked frame) sits mostly there, total does not, so the two
                    ladders are clustered on different transforms.  silhouette_log redoes the
                    search on log(score) for comparison.

    python -m src.cell_activity_silhouette --config recordings.json --out-dir .../silhouette_debug

Writes summary.csv, silhouette_by_k.csv and bootstrap.csv plus one figure per fit.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture

from src.cell_activity import (MIN_PRESENCE, N_LEVELS, NOISE_FLOOR, NOISE_PEAK,
                               activity_descriptors, load_window, save_figure)

K_MAX = 6
N_SEEDS = 20
N_BOOTSTRAP = 200


def ranked_scores(recordings):
    """{fit name: scores} for every ladder a run fits: the cells above the blip threshold."""
    fits, pooled = {}, []
    for name, rec in recordings.items():
        signals, present, _, _ = load_window(rec['csv'], rec['t_start'], rec['t_end'],
                                             rec.get('min_presence', MIN_PRESENCE))
        desc = activity_descriptors(signals, present, rec.get('noise_floor', NOISE_FLOOR))
        desc = desc[desc['peak'] >= rec.get('noise_peak', NOISE_PEAK)]
        fits[f'{name}__total'] = desc['total'].to_numpy()
        pooled.append(desc['rate'].to_numpy())
    fits['pooled__rate'] = np.concatenate(pooled)
    return fits


def kmeans(X, k, seed=0):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)


def silhouettes(X, ks, seed=0):
    out = {}
    for k in ks:
        labels = kmeans(X, k, seed)
        if 1 < len(np.unique(labels)) < len(X):
            out[k] = silhouette_score(X, labels)
    return out


def best_k(scores):
    return max(scores, key=scores.get) if scores else np.nan


def gmm_bic_k(X, k_max):
    bic = {k: GaussianMixture(k, random_state=0, n_init=5).fit(X).bic(X)
           for k in range(1, min(k_max, len(X) - 1) + 1)}
    return min(bic, key=bic.get), bic


def cuts(x, labels):
    """Boundaries between consecutive clusters of 1-D x, in x units."""
    order = sorted(np.unique(labels), key=lambda l: x[labels == l].mean())
    return [(x[labels == a].max() + x[labels == b].min()) / 2 for a, b in zip(order, order[1:])]


def debug_fit(fit, score, out_dir, rng):
    x = np.log1p(score)
    X = x.reshape(-1, 1)
    k_hi = min(K_MAX, len(np.unique(x)) - 1)
    ks = range(2, k_hi + 1)

    by_seed = pd.DataFrame([{'k': k, 'seed': s, 'silhouette': v}
                            for s in range(N_SEEDS) for k, v in silhouettes(X, ks, s).items()])
    by_k = by_seed.groupby('k')['silhouette'].agg(['min', 'max'])
    by_k['seed0'] = pd.Series(silhouettes(X, ks))
    by_k['log'] = pd.Series(silhouettes(np.log(score).reshape(-1, 1), ks))
    by_k.insert(0, 'fit', fit)

    run_ks = [k for k in ks if k <= N_LEVELS]  # what activity_levels searches
    run = {k: by_k.loc[k, 'seed0'] for k in run_ks}
    chosen = best_k(run)
    ranked = sorted(run.values(), reverse=True)
    boot = []
    for _ in range(N_BOOTSTRAP):
        sample = X[rng.integers(0, len(X), len(X))]
        boot.append(best_k(silhouettes(sample, [k for k in run_ks if k < len(np.unique(sample))])))
    boot = pd.Series(boot).value_counts(normalize=True).rename('frac').rename_axis('k')

    bic_k, bic = gmm_bic_k(X, K_MAX)
    row = {'fit': fit, 'n_cells': len(x), 'chosen_k': chosen,
           'best_silhouette': ranked[0] if ranked else np.nan,
           'margin_to_runner_up': ranked[0] - ranked[1] if len(ranked) > 1 else np.nan,
           'max_seed_spread': float((by_k['max'] - by_k['min']).loc[run_ks].max()),
           'bootstrap_frac_chosen': float(boot.get(chosen, 0.0)),
           'bootstrap_mode_k': int(boot.idxmax()),
           'best_k_up_to_6': best_k(by_k['seed0'].to_dict()),
           'best_k_log': best_k(by_k['log'].loc[run_ks].to_dict()),
           'gmm_bic_k': bic_k,
           'frac_score_lt_1': float((score < 1).mean()),
           'score_min': float(score.min()), 'score_median': float(np.median(score)),
           'score_max': float(score.max())}
    row.update({f'boot_k{k}': float(boot.get(k, 0.0)) for k in run_ks})

    plot_fit(fit, x, X, by_k, run_ks, chosen, boot, bic, os.path.join(out_dir, fit))
    return row, by_k.reset_index(), boot.reset_index().assign(fit=fit)


def plot_fit(fit, x, X, by_k, run_ks, chosen, boot, bic, path_stem):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax = axes[0, 0]
    ax.fill_between(by_k.index, by_k['min'], by_k['max'], alpha=0.25, label=f'{N_SEEDS} seeds')
    ax.plot(by_k.index, by_k['seed0'], marker='o', label='seed 0 (log1p, what the run uses)')
    ax.plot(by_k.index, by_k['log'], marker='s', ls='--', label='log instead of log1p')
    ax.axvspan(1.5, N_LEVELS + 0.5, color='grey', alpha=0.08)
    ax.set(xlabel='k', ylabel='silhouette', title=f'silhouette by k (shaded = searched, '
                                                   f'chosen {chosen})')
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.bar(boot.index.astype(str), boot.values)
    ax.set(xlabel='winning k', ylabel='fraction of resamples',
           title=f'bootstrap choice ({N_BOOTSTRAP} resamples)')

    ax = axes[0, 2]
    ax.plot(list(bic), list(bic.values()), marker='o')
    ax.set(xlabel='k', ylabel='BIC (lower is better)', title='Gaussian mixture BIC, k from 1')

    for ax, k in zip(axes[1], sorted({2, 3, chosen, 4} & set(run_ks))[:3]):
        labels = kmeans(X, k)
        ax.hist(x, bins=40, color='#9a988f')
        for c in cuts(x, labels):
            ax.axvline(c, color='red', lw=1.5)
        s = silhouette_samples(X, labels)
        ax.set(xlabel='log1p(score)', ylabel='cells',
               title=f'k={k}: cuts at score {", ".join(f"{np.expm1(c):.3g}" for c in cuts(x, labels))}'
                     f'\nmean silhouette {s.mean():.3f}, {(s < 0).sum()} cells < 0')
    for ax in axes[1][len(sorted({2, 3, chosen, 4} & set(run_ks))[:3]):]:
        ax.axis('off')
    fig.suptitle(f'{fit}: {len(x)} cells')
    fig.tight_layout()
    save_figure(fig, path_stem, eps=False)


def run(recordings, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, by_k, boots = [], [], []
    for fit, score in ranked_scores(recordings).items():
        print(f'{fit}: {len(score)} cells')
        row, k_table, boot = debug_fit(fit, score, out_dir, rng)
        rows.append(row), by_k.append(k_table), boots.append(boot)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    pd.concat(by_k).to_csv(os.path.join(out_dir, 'silhouette_by_k.csv'), index=False)
    pd.concat(boots).to_csv(os.path.join(out_dir, 'bootstrap.csv'), index=False)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='json of {name: recording dict}')
    p.add_argument('--out-dir', required=True, help='tables and one figure per fit go here')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        recordings = json.load(f)
    pd.set_option('display.width', 250)
    print(run(recordings, args.out_dir).round(3).T.to_string())


if __name__ == '__main__':
    main()
