#!/usr/bin/env python3
"""Settle three things the config does not state, by measurement.

1. STEP RATE. mic_0 is 800 samples wide. If the audio is 16 kHz the buffer runs
   at 20 Hz; if 48 kHz, at 60 Hz. Nothing records which. But end-effector speed
   is a physical quantity: a hand wiping a surface moves at roughly 0.05-0.5 m/s.
   Computing speed under both hypotheses tells us which one is real.

2. WHICH MIC IS THE CONTACT MIC. The paper describes a contact mic plus an
   ambient one. A piezo on the tool should show more level and a different
   spectral balance than a mic listening to the room.

3. AUDIO SCALE. The arrays are float64 with tiny first values. Our pipeline
   consumes int16, so we need the real dynamic range before converting.
"""
import os
import pathlib

import numpy as np
import zarr

try:
    import imagecodecs.numcodecs as inc
    inc.register_codecs()
except Exception:                                          # noqa: BLE001
    pass

from paths import ZIP


def main():
    store = zarr.ZipStore(str(ZIP), mode='r')
    root = zarr.open(store, mode='r')
    ends = np.asarray(root['meta/episode_ends'][:]).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]])

    print('=' * 84)
    print('1. STEP RATE, from end-effector speed')
    print('=' * 84)
    speeds = {20.0: [], 60.0: []}
    for e in range(0, min(12, len(ends))):
        s, t = int(starts[e]), int(ends[e])
        pos = np.asarray(root['data/robot0_eef_pos'][s:t], dtype=np.float64)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)     # metres per step
        for hz in (20.0, 60.0):
            speeds[hz].append(np.median(step) * hz)
    for hz in (20.0, 60.0):
        v = np.array(speeds[hz])
        audio_rate = int(hz * 800)
        print(f'  if {audio_rate:5d} Hz audio -> {hz:4.0f} Hz steps: '
              f'median EE speed {v.mean():.3f} m/s  (per-episode range '
              f'{v.min():.3f}-{v.max():.3f})')
    print('\n  Hand-guided wiping is roughly 0.05-0.5 m/s. Pick the plausible one.')

    # Episode duration under each hypothesis.
    lens = np.diff(np.concatenate([[0], ends]))
    print(f'\n  median episode is {int(np.median(lens))} steps = '
          f'{np.median(lens)/20:.1f} s at 20 Hz, or {np.median(lens)/60:.1f} s at 60 Hz')

    print()
    print('=' * 84)
    print('2. WHICH MIC IS WHICH')
    print('=' * 84)
    for name in ('data/mic_0', 'data/mic_1'):
        a = root[name]
        # Sample a few whole episodes rather than the first rows, which are
        # usually before contact starts.
        chunks = []
        for e in (0, 5, 10, 20, 40):
            if e >= len(ends):
                continue
            s, t = int(starts[e]), int(ends[e])
            chunks.append(np.asarray(a[s:min(s + 400, t)], dtype=np.float64).reshape(-1))
        x = np.concatenate(chunks)
        rms = float(np.sqrt((x ** 2).mean()))
        peak = float(np.abs(x).max())
        # Crude spectral balance: high-band energy fraction.
        X = np.abs(np.fft.rfft(x[:2 ** 18] * np.hanning(min(2 ** 18, len(x))))) ** 2
        f = np.fft.rfftfreq(min(2 ** 18, len(x)), 1 / 16000)
        lo = X[(f > 20) & (f < 500)].sum()
        hi = X[(f > 2000) & (f < 8000)].sum()
        print(f'  {name:14s} peak {peak:.5f}  rms {rms:.6f}  '
              f'peak/rms {peak/max(rms,1e-12):6.1f}  hi/lo {10*np.log10(hi/max(lo,1e-20)):+6.1f} dB')

    print()
    print('=' * 84)
    print('3. AUDIO SCALE -> int16')
    print('=' * 84)
    for name in ('data/mic_0', 'data/mic_1'):
        a = root[name]
        s, t = int(starts[0]), int(ends[0])
        x = np.asarray(a[s:t], dtype=np.float64).reshape(-1)
        peak = float(np.abs(x).max())
        print(f'  {name:14s} episode 0 peak {peak:.6f} '
              f'-> x32768 = {peak*32768:8.1f} counts')
    print('\n  buffer_to_fbank peak-normalises each window, so absolute scale does')
    print('  not change the features -- but it does decide int16 quantisation loss.')

    store.close()


if __name__ == '__main__':
    main()
