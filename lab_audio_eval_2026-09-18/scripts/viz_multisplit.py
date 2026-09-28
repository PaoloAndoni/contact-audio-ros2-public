#!/usr/bin/env python3
"""The paired-split figure: does contact audio add anything over proprioception?

Deliberately plots the PAIRED per-split differences rather than four bars of
absolute loss. Split-to-split variation (val MSE swings from 0.63 to 0.81
depending on which episodes are held out) is several times larger than the
effect being tested, so absolute bars would hide the answer. Within a split
every variant sees identical episodes, so the difference is the evidence.
"""
import json
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paths import OUT
from paths import FIG

SURFACE, INK, INK2, INK3, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8a85', '#e3e2dd'
C = {'pose': '#2a78d6', 'audio': '#eb6834', 'both': '#1baf7a', 'mean': '#8a8a85'}
LAB = {'pose': 'pose only', 'audio': 'audio only', 'both': 'audio + pose',
       'mean': 'predict mean (floor)'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'legend.frameon': False,
})


def main():
    d = json.loads((OUT / 'maniwav_multisplit.json').read_text())
    rows = d['rows']
    n = len(rows)
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5),
                                   gridspec_kw={'width_ratios': [1.25, 1]})

    # --- left: absolute loss per split ---------------------------------
    for k in ('mean', 'pose', 'audio', 'both'):
        v = [r[k] for r in rows]
        ax1.plot(x, v, marker='o', ms=5, lw=1.8, color=C[k], label=LAB[k],
                 mec=SURFACE, mew=1.2)
    ax1.set_xlabel('split (which 16 of 78 episodes are held out)')
    ax1.set_ylabel('validation MSE')
    ax1.set_xticks(x)
    ax1.set_title('Loss swings more between splits than between variants')
    ax1.legend(fontsize=8, ncol=2, loc='upper right')

    # --- right: paired differences -------------------------------------
    pb = np.array([r['pose_minus_both'] for r in rows])
    pa = np.array([r['pose_minus_audio'] for r in rows])
    for i, (vals, colour, label, off) in enumerate(
            ((pb, C['both'], 'pose − (audio+pose)', -0.16),
             (pa, C['audio'], 'pose − audio', +0.16))):
        ax2.scatter(x + off, vals, s=34, color=colour, zorder=3,
                    edgecolors=SURFACE, linewidths=1.2, label=label)
        m, sd = vals.mean(), vals.std(ddof=1)
        ax2.axhline(m, color=colour, lw=1.4, ls='--', alpha=0.9, zorder=2)
        ax2.fill_between([-0.6, n - 0.4], m - sd, m + sd, color=colour,
                         alpha=0.10, lw=0, zorder=1)
    ax2.axhline(0, color=INK, lw=1.2, zorder=4)
    ax2.set_xlim(-0.6, n - 0.4)
    ax2.set_xticks(x)
    ax2.set_xlabel('split')
    ax2.set_ylabel('MSE difference  (positive = audio helps)')
    ax2.set_title('Paired difference within each split')
    ax2.legend(fontsize=8, loc='lower right')

    mb, sb = pb.mean(), pb.std(ddof=1)
    ma, sa = pa.mean(), pa.std(ddof=1)
    wins_b = int((pb > 0).sum())
    wins_a = int((pa > 0).sum())
    ax2.text(0.02, 0.97,
             f'pose − (audio+pose)  {mb:+.4f} ± {sb:.4f}   '
             f'helps in {wins_b}/{n}\n'
             f'pose − audio            {ma:+.4f} ± {sa:.4f}   '
             f'helps in {wins_a}/{n}',
             transform=ax2.transAxes, va='top', fontsize=8, color=INK2,
             family='DejaVu Sans Mono')

    fig.suptitle('ManiWAV `wipe`: does contact audio add to proprioception?',
                 fontsize=12, fontweight='bold', x=0.005, ha='left')
    fig.text(0.005, 0.015,
             'Frozen AST encoder (ManiWAV’s pretrained-encoder ablation, not their '
             'headline result). 78 episodes, 8 random episode-level splits, 40 epochs each.\n'
             'Audio alone is worse than pose in 0/8 splits. Adding audio to pose moves the '
             'loss by less than the spread across splits — no measurable benefit here.',
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.085, 1, 0.94))
    fig.savefig(FIG / 'fig7_maniwav_multisplit.png', dpi=300)
    plt.close(fig)
    print('fig7_maniwav_multisplit.png')


if __name__ == '__main__':
    main()
