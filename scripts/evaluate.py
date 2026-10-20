"""Table I (LRE / LMD, mean ± std over samples) and the error vs. azimuth plot (as in Fig. 2b) for the 44.1 kHz test.

Usage:
    python scripts/evaluate.py <test_set_dir> [--proposed hrtf_estimates] [--baselines baselines]

Expects in <test_set_dir>:
    hrtf/<sample>.npy                      ground truth saved by hrtf_estimation.py
    <proposed>/<sample>.npy                proposed estimates (normalised features, de-normalised here)
    <baselines>/{random,generic,nearest_neighbour}/<sample>.npy   from downstream_tasks/baselines.py
    doa/<sample>.np                        DoA (for the azimuth plot)

Metrics are those of downstream_tasks/metrics.py: Table I averages over both ears and all frequencies;
the azimuth plot uses the ipsilateral ear and 6-degree bins.
"""
import os
import sys
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from downstream_tasks.metrics import pointwise_relative_error_dB, pointwise_log_magnitude_distance_dB, mean_std

# normalisation statistics of the prior's training set (hrtf_mu / hrtf_sigma in the config)
MU = np.array([[0.93417883], [0.93417883], [-0.05599184], [-0.05599184]])
SIGMA = np.array([[1.62421620], [1.62421620], [1.35952938], [1.35952938]])
PAPER = {'Random': (-3.43, 1.59, 4.58, 0.69), 'Generic': (-4.01, 1.75, 4.77, 0.74),
         'Proposed': (-9.20, 5.64, 2.28, 1.11), 'Nearest Neighbour': (-7.50, 1.46, 3.76, 0.61)}


def to_complex(x):
    return x[..., :2, :] + 1j * x[..., 2:, :]


def main():
    parser = ArgumentParser()
    parser.add_argument('test_dir')
    parser.add_argument('--proposed', default='hrtf_estimates')
    parser.add_argument('--baselines', default='baselines')
    parser.add_argument('--out_dir', default=None, help='where to write results (default: <test_dir>/evaluation)')
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.test_dir, 'evaluation')
    os.makedirs(out_dir, exist_ok=True)

    methods = {'Random': os.path.join(args.test_dir, args.baselines, 'random'),
               'Generic': os.path.join(args.test_dir, args.baselines, 'generic'),
               'Proposed': os.path.join(args.test_dir, args.proposed),
               'Nearest Neighbour': os.path.join(args.test_dir, args.baselines, 'nearest_neighbour')}

    samples = sorted(os.path.basename(f)[:-4] for f in glob(os.path.join(methods['Proposed'], '*.npy')))
    print(f'{len(samples)} samples with a proposed estimate')

    rows = []
    for sample in samples:
        gt = to_complex(np.load(os.path.join(args.test_dir, 'hrtf', sample + '.npy')))
        doa = np.load(os.path.join(args.test_dir, 'doa', sample + '.np'))
        azimuth = np.degrees(np.arctan2(doa[1], doa[0]))
        ipsilateral = np.argmax((np.abs(gt) ** 2).sum(-1))
        for method, directory in methods.items():
            x = np.load(os.path.join(directory, sample + '.npy'))
            if method == 'Proposed':
                x = x * SIGMA + MU
            x = to_complex(x)
            lre = pointwise_relative_error_dB(x, gt)
            lmd = pointwise_log_magnitude_distance_dB(x, gt)
            rows.append(dict(sample=sample, method=method, azimuth=azimuth, lre=lre.mean(), lmd=lmd.mean(),
                             lre_ipsi=lre[ipsilateral].mean(), lmd_ipsi=lmd[ipsilateral].mean()))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out_dir, 'per_sample.csv'), index=False)

    lines = ['| Method | LRE (ours) | LMD (ours) | LRE (paper) | LMD (paper) |', '|---|---|---|---|---|']
    for method in methods:
        m = d[d.method == method]
        lre, lre_s = mean_std(m.lre.to_numpy())
        lmd, lmd_s = mean_std(m.lmd.to_numpy())
        p = PAPER[method]
        lines.append(f'| {method} | {lre:.2f} ± {lre_s:.2f} | {lmd:.2f} ± {lmd_s:.2f} | {p[0]:.2f} ± {p[1]:.2f} | {p[2]:.2f} ± {p[3]:.2f} |')
    table = '\n'.join(lines)
    print(table)
    open(os.path.join(out_dir, 'table1.md'), 'w').write(f'{len(samples)} samples\n\n' + table + '\n')

    plot_azimuth(d, out_dir)


def plot_azimuth(d, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # lateral angle: 0 = median plane (front/back), ±90 = interaural axis; front and back folded together
    d = d.copy()
    az = np.radians(d.azimuth)
    lateral = np.degrees(np.arctan2(np.sin(az), np.abs(np.cos(az))))
    d['bin'] = (np.floor(lateral / 6) * 6 + 3).clip(-87, 87)
    styles = {'Proposed': ('C3', '-'), 'Generic': ('C2', '--'), 'Nearest Neighbour': ('C0', ':'), 'Random': ('0.5', '-.')}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, metric, label in [(axes[0], 'lre_ipsi', 'LRE (dB)'), (axes[1], 'lmd_ipsi', 'LMD (dB)')]:
        for method, (color, ls) in styles.items():
            g = d[d.method == method].groupby('bin')[metric]
            mean, std = g.mean(), g.std()
            ax.plot(mean.index, mean.values, color=color, ls=ls, lw=2, label=method)
            if method == 'Proposed':
                ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel('lateral angle (deg); 0 = median plane, ±90 = side')
        ax.set_ylabel(label + ', ipsilateral ear')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'error_vs_azimuth.png'), dpi=120)


if __name__ == '__main__':
    main()
