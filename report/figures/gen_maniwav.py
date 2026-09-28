#!/usr/bin/env python3
"""Rebuild the report's ManiWAV figure from saved results, without retraining."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLORS = {'pose': '#2a78d6', 'audio': '#eb6834', 'both': '#1baf7a', 'mean': '#8a8a85'}
LABELS = {'pose': 'Pose only', 'audio': 'Audio only', 'both': 'Audio + pose',
          'mean': 'Full-dataset mean'}
MARKERS = {'pose': 'o', 'audio': 's', 'both': '^', 'mean': 'D'}


def main():
    data = json.loads((ROOT / 'lab_audio_eval_2026-09-18/maniwav_multisplit.json').read_text())
    rows = data['rows']
    x = np.array([r['split'] for r in rows])
    n = len(rows)
    pb = np.array([r['pose'] - r['both'] for r in rows])
    pa = np.array([r['pose'] - r['audio'] for r in rows])
    np.testing.assert_allclose(pb, [r['pose_minus_both'] for r in rows])
    np.testing.assert_allclose(pa, [r['pose_minus_audio'] for r in rows])

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#d6d6d6', 'axes.labelcolor': '#404040',
        'xtick.color': '#404040', 'ytick.color': '#404040',
        'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': '#e5e5e5', 'grid.linewidth': 0.7,
        'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'legend.frameon': False, 'pdf.fonttype': 42,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6.3))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.37, top=0.76, wspace=0.30)
    fig.suptitle('ManiWAV wipe: does audio add to pose?', x=0.02, y=0.98,
                 ha='left', fontsize=16, fontweight='bold')
    fig.text(0.02, 0.92, 'Frozen AST embeddings | 78 episodes | 16 held out per split',
             color='#555555', fontsize=11)

    for key in ('mean', 'pose', 'audio', 'both'):
        left.plot(x, [r[key] for r in rows], marker=MARKERS[key], markersize=5,
                  linewidth=1.8, color=COLORS[key], label=LABELS[key],
                  markeredgecolor='white', markeredgewidth=0.8)
    left.set_title('Validation loss by split', loc='left', pad=12)
    left.set_ylabel('MSE (normalized action targets)')
    left.set_ylim(0.60, 1.15)
    left.legend(loc='lower left', bbox_to_anchor=(-0.015, 1.18), ncol=2,
                fontsize=9, borderaxespad=0, columnspacing=1.5)

    for vals, key, label, offset in (
        (pb, 'both', 'Pose - (audio + pose)', -0.13),
        (pa, 'audio', 'Pose - audio', 0.13),
    ):
        mean, sd = vals.mean(), vals.std(ddof=1)
        right.axhspan(mean - sd, mean + sd, color=COLORS[key], alpha=0.11, lw=0)
        right.axhline(mean, color=COLORS[key], ls='--', lw=1.6)
        right.scatter(x + offset, vals, s=43, marker=MARKERS[key],
                      color=COLORS[key], edgecolors='white', linewidths=0.8,
                      zorder=3, label=label)
    right.axhline(0, color='#222222', lw=1.1)
    right.set_title('Paired difference within each split', loc='left', pad=12)
    right.set_ylabel('Pose MSE - audio configuration MSE')
    right.set_ylim(-0.115, 0.045)
    right.legend(loc='lower left', bbox_to_anchor=(-0.015, 1.18), ncol=2,
                 fontsize=9, borderaxespad=0, columnspacing=1.0)
    for ax in (left, right):
        ax.set_xticks(x)
        ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
        ax.set_xlabel('Episode split')

    # Keep the summary outside the plot area so it cannot obscure data.
    fig.text(0.075, 0.255, 'Paired MSE difference: mean +/- sample SD',
             fontsize=11, fontweight='bold')
    for y, vals, key, label in (
        (0.21, pb, 'both', 'Pose - (audio + pose)'),
        (0.165, pa, 'audio', 'Pose - audio'),
    ):
        fig.text(0.075, y, label, color=COLORS[key], fontsize=11, fontweight='bold')
        fig.text(0.365, y, f'{vals.mean():+.4f} +/- {vals.std(ddof=1):.4f}', fontsize=11)
        fig.text(0.66, y, f'Audio configuration better in {int((vals > 0).sum())}/{n} splits',
                 fontsize=11)
    fig.text(0.075, 0.103, 'Positive differences favor the audio configuration. '
             'Dashed lines: means; shaded bands: +/-1 SD.', fontsize=9, color='#555555')
    fig.text(0.075, 0.065, f'Exploratory scores: best validation epoch of '
             f'{data["config"]["epochs"]}; normalization uses all episodes.',
             fontsize=9, color='#555555')
    fig.text(0.075, 0.027, f'Audio alone is worse in {int((pa < 0).sum())}/{n} splits. '
             'The added value of audio alongside pose remains inconclusive.',
             fontsize=9, color='#555555')
    for extension in ('.png', '.pdf'):
        path = Path(__file__).with_name('maniwav_multisplit' + extension)
        fig.savefig(path, dpi=300, facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
