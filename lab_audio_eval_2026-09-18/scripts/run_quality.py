#!/usr/bin/env python3
"""Run the per-clip quality pass over every lab bag and save a CSV + report."""
import csv
import json
import os
import pathlib

import numpy as np

import audio_io
import quality

from paths import BAGS
from paths import OUT
OUT.mkdir(parents=True, exist_ok=True)


def parse_name(name):
    """table_cleaning_noiseoff_01 -> (action, noise, take)."""
    parts = name.split('_')
    take = parts[-1]
    noise = parts[-2]
    action = '_'.join(parts[:-2])
    return action, noise, take


def main():
    rows = []
    for d in sorted(p for p in BAGS.iterdir() if p.is_dir()):
        L = audio_io.load_bag(d)
        mono = audio_io.to_mono_float(L)
        q = quality.analyse(L, mono)
        v, reasons = quality.verdict(q)
        action, noise, take = parse_name(d.name)
        rows.append({
            'clip': d.name, 'action': action, 'noise': noise, 'take': take,
            'sample_rate': q['sample_rate'], 'channels': q['channels'],
            'duration_s': round(q['duration_s'], 3),
            'n_chunks': q['n_chunks'],
            'gaps': q['gap_count'], 'gap_ms': round(q['gap_seconds'] * 1000, 1),
            'overlap_frames': q['overlap_frames'],
            'peak_counts': int(q['peak_counts']),
            'peak_dbfs': round(q['peak_dbfs'], 1),
            'used_bits': round(q['used_bits'], 1),
            'rms_dbfs': round(q['rms_dbfs'], 1),
            'pct_clipped': round(q['pct_clipped'], 4),
            'floor_dbfs': round(q['floor_dbfs'], 1),
            'n_events': q['n_events'],
            'event_rms_dbfs': round(q['event_rms_dbfs'], 1),
            'event_time_s': round(q['event_time_s'], 2),
            'snr_db': round(q['snr_db'], 1),
            'peak_snr_db': round(q['peak_snr_db'], 1),
            'verdict': v,
            'why': '; '.join(reasons),
        })
        print(f'{d.name:34s} {v:9s} peak={int(q["peak_counts"]):6d}cts '
              f'({q["peak_dbfs"]:6.1f} dBFS)  rms={q["rms_dbfs"]:6.1f}  '
              f'floor={q["floor_dbfs"]:6.1f}  SNR={q["snr_db"]:5.1f} dB  '
              f'events={q["n_events"]:3d}  gaps={q["gap_count"]}')

    fields = list(rows[0].keys())
    with open(OUT / 'clip_quality.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / 'clip_quality.json', 'w') as f:
        json.dump(rows, f, indent=2)

    print()
    print('=' * 70)
    peaks = np.array([r['peak_counts'] for r in rows])
    snrs = np.array([r['snr_db'] for r in rows], dtype=float)
    print(f'peak counts  min={peaks.min()}  median={np.median(peaks):.0f}  max={peaks.max()}')
    print(f'SNR dB       min={np.nanmin(snrs):.1f}  median={np.nanmedian(snrs):.1f}  '
          f'max={np.nanmax(snrs):.1f}')
    from collections import Counter
    print('verdicts:', dict(Counter(r['verdict'] for r in rows)))
    print(f'\nwrote {OUT}/clip_quality.csv')


if __name__ == '__main__':
    main()
