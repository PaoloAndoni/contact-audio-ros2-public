#!/usr/bin/env python3
"""Publication-quality figures for the 18 Sept 2026 contact-audio session.

Palette: dataviz categorical slots 1-3 (blue / orange / aqua), which validate
all-pairs for both normal and CVD vision at three series. Action identity is
carried by colour AND marker shape, so it never depends on colour alone.
Spectrograms use magma: luminance-monotonic and CVD-safe, not a rainbow.
"""
import json
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import audio_io
import quality

from paths import OUT
from paths import FIG
FIG.mkdir(parents=True, exist_ok=True)
from paths import BAGS
SESSION = os.environ.get('SESSION_LABEL', '18 Sept 2026 lab session')

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
INK3 = '#8a8a85'
GRID = '#e3e2dd'
ACTION_COLOR = {'put_sizes': '#2a78d6', 'red_green': '#eb6834',
                'table_cleaning': '#1baf7a'}
ACTION_MARKER = {'put_sizes': 'o', 'red_green': 's', 'table_cleaning': '^'}
ACTION_LABEL = {'put_sizes': 'put_sizes', 'red_green': 'red_green',
                'table_cleaning': 'table_cleaning'}
STATUS = {'USABLE': '#1baf7a', 'MARGINAL': '#eda100', 'UNUSABLE': '#e34948'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False,
})


def parse_name(n):
    p = n.split('_')
    return '_'.join(p[:-2]), p[-2], p[-1]


# ---------------------------------------------------------------- figure 1
def fig_levels(rows):
    """One glance at why this session failed: every clip against full scale."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    rows = sorted(rows, key=lambda r: (r['action'], r['noise'], r['take']))
    y = np.arange(len(rows))
    floor = np.array([r['floor_dbfs'] for r in rows])
    peak = np.array([r['peak_dbfs'] for r in rows])

    for i, r in enumerate(rows):
        c = ACTION_COLOR[r['action']]
        ax.plot([floor[i], peak[i]], [i, i], color=c, lw=2, solid_capstyle='round',
                zorder=2)
        ax.plot(peak[i], i, marker=ACTION_MARKER[r['action']], ms=6, color=c,
                mec=SURFACE, mew=1.2, zorder=3)
        ax.plot(floor[i], i, marker='|', ms=7, color=INK3, zorder=3)

    # Reference bands: where usable contact audio should sit.
    ax.add_patch(Rectangle((-18, -1), 18, len(rows) + 1, color='#1baf7a',
                           alpha=0.07, zorder=0, lw=0))
    ax.text(-9, len(rows) - 0.3, 'healthy peak range', color='#0f7a55',
            fontsize=8, ha='center', va='center')

    ax.set_yticks(y)
    ax.set_yticklabels([r['clip'] for r in rows], fontsize=8)
    ax.set_xlabel('level (dBFS)   —   bar spans noise floor to peak')
    ax.set_xlim(-80, 0)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    lo_db, hi_db = float(np.min(-peak)), float(np.max(-peak))
    ax.set_title(f'Peak level per clip — {SESSION}')
    ax.grid(axis='y', visible=False)

    handles = [Line2D([], [], color=ACTION_COLOR[a], marker=ACTION_MARKER[a],
                      lw=2, ms=6, mec=SURFACE, label=ACTION_LABEL[a])
               for a in sorted(ACTION_COLOR)]
    handles.append(Line2D([], [], color=INK3, marker='|', lw=0, label='noise floor'))
    ax.legend(handles=handles, loc='lower right', fontsize=8, ncol=2)
    pk_counts = np.array([r['peak_counts'] for r in rows], dtype=float)
    fig.text(0.01, 0.01,
             f'Loudest clip peaks at {peak.max():.1f} dBFS; median clip at '
             f'{np.median(peak):.1f} dBFS '
             f'({np.median(pk_counts):.0f} of 32768 counts, '
             f'~{np.log2(np.median(pk_counts)+1):.1f} of 16 bits).',
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / 'fig1_levels.png', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_clip_panels(clip_names, disp, out_name='fig2_clip_panels.png', title=None):
    """Waveform + level-faithful log-mel spectrogram for representative clips."""
    n = len(clip_names)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 6.0),
                             gridspec_kw={'height_ratios': [1, 1.35]})
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, name in enumerate(clip_names):
        L = audio_io.load_bag(BAGS / name)
        mono = audio_io.to_mono_float(L)
        raw = L['data'][:, 0].astype(np.float64)
        sr = L['sample_rate']
        action, noise, take = parse_name(name)
        col = ACTION_COLOR[action]
        t = np.arange(len(raw)) / sr

        # --- waveform, in raw int16 counts so the failure is legible ---
        ax = axes[0, j]
        ax.plot(t, raw, color=col, lw=0.35)
        sd = raw.std()
        ax.axhspan(-6 * sd, 6 * sd, color=INK3, alpha=0.16, lw=0,
                   label='±6σ of the whole clip')
        q = quality.analyse(L, mono)
        v, _ = quality.verdict(q)
        lim = max(np.abs(raw).max() * 1.15, 8 * sd)
        ax.set_ylim(-lim, lim)
        ax.set_xlim(0, t[-1])
        ax.set_ylabel('amplitude (int16 counts)' if j == 0 else '')
        ax.set_title(f'{name}\n{v} · peak {int(np.abs(raw).max())} cts '
                     f'({q["peak_dbfs"]:.1f} dBFS) · clip σ = {sd:.1f} cts',
                     fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        # Full scale for reference, when it would otherwise be invisible.
        ax.text(0.01, 0.96, f'full scale = ±32768 cts ({lim/32768*100:.2f}% shown)',
                transform=ax.transAxes, fontsize=7, color=INK3, va='top')

        # --- log-mel spectrogram ---
        ax = axes[1, j]
        fb = disp[name]                      # (frames, 128), natural log power
        db = fb.T * (10.0 / np.log(10.0))    # -> dB
        # Common scale across panels: anchored to the quietest clip's floor.
        vmin = float(os.environ.get("SPEC_VMIN", -68))
        vmax = float(os.environ.get("SPEC_VMAX", -30))
        im = ax.imshow(db, aspect='auto', origin='lower', cmap='magma',
                       vmin=vmin, vmax=vmax,
                       extent=(0, fb.shape[0] * 0.01, 0, 128))
        ax.set_xlabel('time (s)')
        ax.set_ylabel('mel bin (128)' if j == 0 else '')
        ax.set_title('log-mel spectrogram'
                     , fontsize=9)
        ax.grid(False)
        for a, b in q['events']:
            ax.axvspan(a * 0.01, b * 0.01, color='#ffffff', alpha=0.0)
            ax.plot([a * 0.01, b * 0.01], [124, 124], color='#4dd4a5', lw=2.5,
                    solid_capstyle='butt')
        if q['events']:
            ax.text(0.01, 0.94, f'{len(q["events"])} detected event(s), marked above',
                    transform=ax.transAxes, fontsize=7, color='#8ff0c4', va='top')
        else:
            ax.text(0.01, 0.94, 'no events above floor + 6 dB',
                    transform=ax.transAxes, fontsize=7, color='#ffb3b3', va='top')
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.035)
        cb.set_label('dB', fontsize=8)
        cb.outline.set_visible(False)

    fig.suptitle(title if title is not None else f'Representative clips — {SESSION}',
                 fontsize=12, fontweight='bold', x=0.005, ha='left')
    fig.tight_layout(rect=(0, 0, 0.985, 0.96))
    fig.savefig(FIG / out_name, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_embedding(meta, sep):
    encs = [e for e in ('cnn', 'melstats', 'ast') if (OUT / f'tsne_{e}.npy').exists()]
    if not encs:
        print('no t-SNE maps yet')
        return
    actions = np.array([m['action'] for m in meta])
    fig, axes = plt.subplots(1, len(encs), figsize=(4.5 * len(encs), 4.8))
    if len(encs) == 1:
        axes = [axes]
    for ax, enc in zip(axes, encs):
        Z = np.load(OUT / f'tsne_{enc}.npy')
        for a in sorted(ACTION_COLOR):
            m = actions == a
            ax.scatter(Z[m, 0], Z[m, 1], s=16, c=ACTION_COLOR[a],
                       marker=ACTION_MARKER[a], linewidths=0.5,
                       edgecolors=SURFACE, alpha=0.85, label=ACTION_LABEL[a])
        s = sep.get(enc, {})
        ax.set_title(
            f'{enc}\nsilhouette {s.get("silhouette_action", float("nan")):+.3f} · '
            f'per-clip LOCO {s.get("loco_clip_acc", float("nan")):.2f} · '
            f'p = {s.get("p_value", float("nan")):.4f}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GRID)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=3, fontsize=9,
               markerscale=1.4, bbox_to_anchor=(0.5, 0.085))
    fig.suptitle('t-SNE of window embeddings by sound category',
                 fontsize=12, fontweight='bold', x=0.005, ha='left')
    n_win = len(meta)
    n_clip = len({m['clip'] for m in meta})
    fig.text(0.005, 0.012,
             f'n = {n_win} one-second windows from {n_clip} clips. Windows within a '
             f'clip are not independent: the effective sample size is {n_clip}, not '
             f'{n_win}.\n'
             'p is a clip-level label-permutation test against leave-one-clip-out '
             'accuracy. It does not test the layout of this plot.',
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(FIG / 'fig3_embedding.png', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_table(rows):
    rows = sorted(rows, key=lambda r: (r['action'], r['noise'], r['take']))
    cols = [('clip', 1.00), ('sr', 0.26), ('ch', 0.16), ('dur (s)', 0.34),
            ('gaps', 0.24), ('peak (cts)', 0.42), ('peak (dBFS)', 0.44),
            ('RMS (dBFS)', 0.44), ('floor (dBFS)', 0.44), ('clip %', 0.30),
            ('events', 0.30), ('SNR (dB)', 0.36), ('verdict', 0.46)]
    widths = np.array([w for _, w in cols])
    widths = widths / widths.sum()
    xs = np.concatenate([[0], np.cumsum(widths)])

    fig_h = 0.30 * len(rows) + 1.5
    fig, ax = plt.subplots(figsize=(12.6, fig_h))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows) + 2)

    yhdr = len(rows) + 0.7
    for (name, _), x0, w in zip(cols, xs[:-1], widths):
        ax.text(x0 + 0.004, yhdr, name, fontsize=8.5, fontweight='bold',
                color=INK, va='center')
    ax.plot([0, 1], [yhdr - 0.45, yhdr - 0.45], color=INK, lw=1.0)

    for i, r in enumerate(rows):
        y = len(rows) - i - 0.3
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y - 0.42), 1, 0.84, color='#f4f3ef',
                                   lw=0, zorder=0))
        snr = r['snr_db']
        vals = [
            r['clip'], f'{r["sample_rate"]//1000} kHz', str(r['channels']),
            f'{r["duration_s"]:.2f}', str(r['gaps']), str(r['peak_counts']),
            f'{r["peak_dbfs"]:.1f}', f'{r["rms_dbfs"]:.1f}',
            f'{r["floor_dbfs"]:.1f}', f'{r["pct_clipped"]:.2f}',
            str(r['n_events']),
            '—' if snr != snr else f'{snr:.1f}',
            r['verdict'],
        ]
        for j, (v, x0) in enumerate(zip(vals, xs[:-1])):
            if cols[j][0] == 'verdict':
                ax.text(x0 + 0.004, y, v, fontsize=8, color=STATUS[r['verdict']],
                        fontweight='bold', va='center', zorder=2)
            elif cols[j][0] == 'clip':
                ax.text(x0 + 0.004, y, v, fontsize=8, color=INK, va='center',
                        zorder=2)
            else:
                ax.text(x0 + 0.004, y, v, fontsize=8, color=INK2, va='center',
                        zorder=2)

    ax.set_title(f'Per-clip audio quality, {SESSION}  '
                 '(all values measured from the decoded bags)',
                 fontsize=11, fontweight='bold', loc='left', pad=14)
    fig.text(0.007, 0.012,
             'SNR = mean event power / noise-floor power, events = runs of ≥3 frames '
             'more than 6 dB above the floor.  “—” = no event detected.  '
             'Verdict thresholds: UNUSABLE if peak < 64 counts, or SNR < 6 dB, or no events.',
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIG / 'fig4_quality_table.png', dpi=300)
    plt.close(fig)


def main():
    rows = json.loads((OUT / 'clip_quality.json').read_text())
    # Reference takes carry no action label; keep them out of the figures.
    rows = [r for r in rows if r['action'] in ACTION_COLOR]
    meta = json.loads((OUT / 'window_meta.json').read_text())
    sep = {}
    if (OUT / 'separability.json').exists():
        sep = json.loads((OUT / 'separability.json').read_text())
    disp = dict(np.load(OUT / 'display_spectrograms.npz'))

    fig_levels(rows)
    print('fig1_levels.png')
    fig_clip_panels(os.environ.get("PANEL_CLIPS", "red_green_noiseoff_03,put_sizes_noiseoff_02,table_cleaning_noiseoff_01").split(","), disp)
    print('fig2_clip_panels.png')
    fig_embedding(meta, sep)
    print('fig3_embedding.png')
    fig_table(rows)
    print('fig4_quality_table.png')
    print(f'\nfigures in {FIG}')


if __name__ == '__main__':
    main()
