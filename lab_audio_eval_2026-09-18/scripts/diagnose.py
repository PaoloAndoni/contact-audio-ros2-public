#!/usr/bin/env python3
"""Work out WHAT the near-silence is: dead channel, quantisation noise, or
a real but tiny acoustic signal.

Three discriminators:
  1. Value histogram. A handful of distinct int16 codes means the ADC saw
     almost nothing; a broad smooth distribution means real analogue noise.
  2. Spectrum shape. Flat = electrical/ADC noise. Tilted with peaks at 50 Hz
     and harmonics, or a broadband rolloff, = something acoustic.
  3. Does noiseon differ from noiseoff? The noise source was deliberately
     switched; a working microphone MUST show it.
"""
import os
import pathlib

import numpy as np

import audio_io

from paths import BAGS


def spectrum(x, sr, nfft=32768):
    n = min(len(x), sr * 8)
    seg = x[:n]
    win = np.hanning(nfft)
    step = nfft // 2
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for s in range(0, len(seg) - nfft + 1, step):
        f = np.fft.rfft(seg[s:s + nfft] * win)
        acc += np.abs(f) ** 2
        cnt += 1
    if cnt == 0:
        return None, None
    psd = acc / cnt
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    return freqs, psd


def main():
    dirs = sorted(p for p in BAGS.iterdir() if p.is_dir())
    print('=' * 100)
    print('1. INT16 VALUE DISTRIBUTION  (how many distinct codes the ADC actually produced)')
    print('=' * 100)
    stash = {}
    for d in dirs:
        L = audio_io.load_bag(d)
        raw = L['data'][:, 0].astype(np.int64)
        stash[d.name] = (L, raw)
        uniq, counts = np.unique(raw, return_counts=True)
        order = np.argsort(-counts)[:5]
        top = ', '.join(f'{uniq[i]}:{100*counts[i]/len(raw):.1f}%' for i in order)
        print(f'{d.name:34s} distinct={len(uniq):6d}  range=[{raw.min()},{raw.max()}]  '
              f'std={raw.std():7.2f}  top codes: {top}')

    print()
    print('=' * 100)
    print('2. SPECTRUM SHAPE  (band power in dB, relative to total)')
    print('=' * 100)
    bands = [(20, 100), (100, 500), (500, 2000), (2000, 8000), (8000, 20000)]
    print(f'{"clip":34s} ' + ' '.join(f'{a}-{b}Hz'.rjust(11) for a, b in bands)
          + '   50Hz/nbr')
    for d in dirs:
        L, raw = stash[d.name]
        mono = audio_io.to_mono_float(L)
        sr = L['sample_rate']
        freqs, psd = spectrum(mono.astype(np.float64), sr)
        if psd is None:
            continue
        tot = psd.sum()
        cells = []
        for a, b in bands:
            m = (freqs >= a) & (freqs < b)
            cells.append(f'{10*np.log10(max(psd[m].sum()/tot,1e-20)):11.1f}')
        # Mains hum test: 50 Hz bin vs its neighbourhood.
        m50 = (freqs > 45) & (freqs < 55)
        mnb = ((freqs > 25) & (freqs < 45)) | ((freqs > 55) & (freqs < 80))
        hum = 10 * np.log10(max(psd[m50].max(), 1e-20) / max(np.median(psd[mnb]), 1e-20))
        print(f'{d.name:34s} ' + ' '.join(cells) + f'   {hum:6.1f} dB')

    print()
    print('=' * 100)
    print('3. DOES THE NOISE SOURCE SHOW UP?  (noiseon vs noiseoff, same action)')
    print('=' * 100)
    groups = {}
    for d in dirs:
        parts = d.name.split('_')
        action, noise = '_'.join(parts[:-2]), parts[-2]
        L, raw = stash[d.name]
        mono = audio_io.to_mono_float(L)
        r = float(np.sqrt((mono.astype(np.float64) ** 2).mean()))
        groups.setdefault((action, noise), []).append((d.name, r, int(np.abs(raw).max())))
    for action in sorted({a for a, _ in groups}):
        print(f'\n  {action}:')
        for noise in ('noiseoff', 'noiseon'):
            g = groups.get((action, noise), [])
            if not g:
                print(f'    {noise:9s} (none)')
                continue
            rmss = np.array([r for _, r, _ in g])
            pks = np.array([p for _, _, p in g])
            print(f'    {noise:9s} n={len(g)}  RMS {20*np.log10(rmss.mean()):7.2f} dBFS'
                  f'   peak counts {pks.tolist()}')

    print()
    print('=' * 100)
    print('4. THE ONE LOUD CLIP  red_green_noiseoff_03')
    print('=' * 100)
    L, raw = stash['red_green_noiseoff_03']
    mono = audio_io.to_mono_float(L)
    sr = L['sample_rate']
    env = np.abs(mono)
    w = sr // 20
    sm = np.convolve(env, np.ones(w) / w, mode='same')
    loud = sm > sm.max() * 0.1
    idx = np.where(loud)[0]
    if len(idx):
        print(f'  loud region: {idx[0]/sr:.2f}s .. {idx[-1]/sr:.2f}s '
              f'of {L["duration_s"]:.2f}s  ({100*len(idx)/len(mono):.1f}% of clip)')
    print(f'  peak {np.abs(raw).max()} counts at t={np.argmax(np.abs(raw))/sr:.2f}s')
    # Is the rest of this clip as dead as the others?
    quiet = mono[~loud] if len(idx) else mono
    print(f'  RMS outside the loud region: '
          f'{20*np.log10(max(np.sqrt((quiet.astype(np.float64)**2).mean()),1e-12)):.1f} dBFS')


if __name__ == '__main__':
    main()
