#!/usr/bin/env python3
"""Figures for the ManiWAV `wipe` audio-policy experiment.

Same palette and conventions as the lab-session figures, so the two sets read
as one report. Three series carry identity (pose / audio / both); the mean
baseline is deliberately NEUTRAL grey -- it is a reference line, not a
competitor, and colouring it like one would imply a fourth condition.
"""
import json
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paths import OUT
from paths import FIG
FIG.mkdir(parents=True, exist_ok=True)

SURFACE = '#fcfcfb'
INK, INK2, INK3, GRID = '#0b0b0b', '#52514e', '#8a8a85', '#e3e2dd'
COLOR = {'pose': '#2a78d6', 'audio': '#eb6834', 'both': '#1baf7a', 'mean': '#8a8a85'}
MARKER = {'pose': 'o', 'audio': 's', 'both': '^', 'mean': 'D'}
LABEL = {'pose': 'pose only', 'audio': 'audio only', 'both': 'audio + pose',
         'mean': 'predict mean (floor)'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False,
})


def fig_curves_and_bars(res):
    variants = [k for k in ('pose', 'audio', 'both') if k in res]
    mean_mse = res['mean']['val_mse']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6),
                                   gridspec_kw={'width_ratios': [1.35, 1]})

    # --- learning curves -------------------------------------------------
    for v in variants:
        h = res[v]['history']
        ep = [r['epoch'] for r in h]
        ax1.plot(ep, [r['val'] for r in h], color=COLOR[v], lw=2,
                 label=LABEL[v], zorder=3)
        ax1.plot(ep, [r['train'] for r in h], color=COLOR[v], lw=1,
                 ls=':', alpha=0.75, zorder=2)
    ax1.axhline(mean_mse, color=COLOR['mean'], lw=1.5, ls='--', zorder=1)
    ax1.text(ax1.get_xlim()[1], mean_mse, '  predict-mean floor', va='center',
             fontsize=8, color=INK2)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('MSE (normalised action units)')
    ax1.set_title('Validation loss, held-out episodes')
    handles = [Line2D([], [], color=COLOR[v], lw=2, label=LABEL[v]) for v in variants]
    handles += [Line2D([], [], color=INK3, lw=1, ls=':', label='train (dotted)')]
    ax1.legend(handles=handles, fontsize=8, loc='upper right')

    # --- final comparison ------------------------------------------------
    keys = ['mean'] + variants
    vals = [res[k]['val_mse'] for k in keys]
    y = np.arange(len(keys))
    for i, k in enumerate(keys):
        ax2.barh(i, res[k]['val_mse'], height=0.55, color=COLOR[k],
                 edgecolor=SURFACE, linewidth=2, zorder=2)
        imp = res[k].get('improvement_vs_mean')
        txt = f'{res[k]["val_mse"]:.4f}'
        if imp is not None:
            txt += f'   ({100*imp:+.1f}% vs floor)'
        ax2.text(res[k]['val_mse'] * 1.02, i, txt, va='center', fontsize=8,
                 color=INK2, zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([LABEL[k] for k in keys], fontsize=9)
    ax2.set_xlim(0, max(vals) * 1.45)
    ax2.set_xlabel('best validation MSE (lower is better)')
    ax2.set_title('Does audio add anything over pose?')
    ax2.grid(axis='y', visible=False)
    ax2.invert_yaxis()

    fig.suptitle('ManiWAV `wipe`: frozen-encoder audio policy',
                 fontsize=12, fontweight='bold', x=0.005, ha='left')
    fig.text(0.005, 0.015,
             'Encoder is FROZEN (ManiWAV\'s pretrained-encoder ablation, not their headline '
             'result). Splits are by episode, never by timestep.\n'
             'The claim to look for is audio or audio+pose beating pose alone — pose by '
             'itself already predicts smooth motion well.',
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.085, 1, 0.94))
    fig.savefig(FIG / 'fig5_maniwav_training.png', dpi=300)
    plt.close(fig)


def fig_action_error(res):
    """Per-channel error, in the units the action actually carries."""
    variants = [k for k in ('pose', 'audio', 'both') if k in res]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    chans = [('trans_err_m', 'translation', 'mm', 1000.0),
             ('rot6d_err', 'rotation (6-D)', 'abs err', 1.0),
             ('grip_err', 'gripper', 'abs err', 1.0)]
    for ax, (key, name, unit, scale) in zip(axes, chans):
        vals = [res[v][key] * scale for v in variants]
        for i, v in enumerate(variants):
            ax.bar(i, res[v][key] * scale, width=0.6, color=COLOR[v],
                   edgecolor=SURFACE, linewidth=2)
            ax.text(i, res[v][key] * scale, f'{res[v][key]*scale:.3g}',
                    ha='center', va='bottom', fontsize=8, color=INK2)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([LABEL[v] for v in variants], fontsize=8, rotation=12)
        ax.set_ylabel(unit)
        ax.set_title(name)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis='x', visible=False)
    fig.suptitle('Held-out action error by channel, 16-step chunk',
                 fontsize=12, fontweight='bold', x=0.005, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / 'fig6_maniwav_action_error.png', dpi=300)
    plt.close(fig)


def main():
    p = OUT / 'maniwav_wipe_results.json'
    if not p.exists():
        raise SystemExit(f'{p} not found - run train_head.py first')
    res = json.loads(p.read_text())
    fig_curves_and_bars(res)
    print('fig5_maniwav_training.png')
    fig_action_error(res)
    print('fig6_maniwav_action_error.png')
    print(f'figures in {FIG}')


if __name__ == '__main__':
    main()
