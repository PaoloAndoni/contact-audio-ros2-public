#!/usr/bin/env python3
"""Are the rare excursions above the 5.8-count floor real contacts or glitches?

Discriminators:
  - Width. An acoustic contact rings for milliseconds; a digital glitch is
    1-3 samples.
  - Chunk-boundary alignment. Chunks are 2048 frames. Excursions landing on
    multiples of 2048 are a buffering artifact, not sound.
  - Envelope shape. A contact has a sharp attack and an exponential decay.
"""
import os
import pathlib

import numpy as np

import audio_io

from paths import BAGS
CHUNK = 2048


def main():
    print(f'{"clip":34s} {"n_exc":>5s} {"widths(samples)":>24s} '
          f'{"max_cts":>8s} {"chunk-aligned":>14s}')
    print('-' * 100)
    details = {}
    for d in sorted(p for p in BAGS.iterdir() if p.is_dir()):
        L = audio_io.load_bag(d)
        raw = L['data'][:, 0].astype(np.int64)
        sr = L['sample_rate']
        sd = raw.std()
        # 6 sigma above the floor: well clear of Gaussian noise over 7e5 samples.
        thr = 6 * sd
        over = np.abs(raw) > thr
        if not over.any():
            print(f'{d.name:34s} {0:5d}')
            details[d.name] = []
            continue
        # Group excursions separated by < 5 ms into one event.
        idx = np.where(over)[0]
        splits = np.where(np.diff(idx) > sr * 0.005)[0]
        groups = np.split(idx, splits + 1)
        widths = [int(g[-1] - g[0] + 1) for g in groups]
        peaks = [int(np.abs(raw[g[0]:g[-1] + 1]).max()) for g in groups]
        # Distance of each event start from the nearest chunk boundary.
        offs = [int(min(g[0] % CHUNK, CHUNK - g[0] % CHUNK)) for g in groups]
        aligned = sum(1 for o in offs if o <= 2)
        wtxt = ','.join(str(w) for w in widths[:6]) + ('...' if len(widths) > 6 else '')
        print(f'{d.name:34s} {len(groups):5d} {wtxt:>24s} {max(peaks):8d} '
              f'{aligned:>6d}/{len(groups):<7d}')
        details[d.name] = [
            {'start': int(g[0]), 't': float(g[0]) / sr, 'width': w,
             'peak': p, 'chunk_off': o}
            for g, w, p, o in zip(groups, widths, peaks, offs)
        ]

    print()
    print('=' * 100)
    print('DETAIL: every excursion, with duration in ms and time in clip')
    print('=' * 100)
    for name, evs in details.items():
        if not evs:
            continue
        L = audio_io.load_bag(BAGS / name)
        sr = L['sample_rate']
        print(f'\n{name}:')
        for e in evs:
            print(f'   t={e["t"]:6.2f}s  width={e["width"]:5d} samp '
                  f'({1000*e["width"]/sr:7.2f} ms)  peak={e["peak"]:5d} cts  '
                  f'dist-to-chunk-boundary={e["chunk_off"]:4d}')

    print()
    print('=' * 100)
    print('ENVELOPE of the single largest excursion (red_green_noiseoff_03)')
    print('=' * 100)
    L = audio_io.load_bag(BAGS / 'red_green_noiseoff_03')
    raw = L['data'][:, 0].astype(np.int64)
    sr = L['sample_rate']
    c = int(np.argmax(np.abs(raw)))
    lo, hi = max(0, c - int(0.004 * sr)), min(len(raw), c + int(0.020 * sr))
    seg = raw[lo:hi]
    # Print a coarse envelope: max |sample| per 1 ms.
    step = sr // 1000
    print(f'  peak sample index {c} (t={c/sr:.4f}s), chunk offset {c % CHUNK}')
    print('  per-ms peak around the event:')
    for i in range(0, len(seg) - step, step):
        t_ms = (lo + i - c) / sr * 1000
        v = int(np.abs(seg[i:i + step]).max())
        bar = '#' * min(60, int(60 * v / max(np.abs(seg).max(), 1)))
        print(f'    {t_ms:+7.1f} ms  {v:6d}  {bar}')


if __name__ == '__main__':
    main()
