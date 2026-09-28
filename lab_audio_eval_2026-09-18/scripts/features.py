#!/usr/bin/env python3
"""Compute pipeline-exact log-mel features and encoder embeddings for every clip.

Two distinct feature products, for two different questions:

  WINDOW FEATURES (pipeline-exact). 1.0 s windows -> buffer_to_fbank ->
  (98, 128) kaldi fbank, peak-normalised per window exactly as the live
  pipeline does. These feed the encoders, so the embedding study sees
  precisely what contact_ml_node would see.

  DISPLAY SPECTROGRAM. Uses compute_fbank on the whole clip, without the
  per-window peak scaling used for encoder input. This keeps changes in
  recorded level visible.
"""
import os
import pathlib

import numpy as np

import paths  # Set the package path before importing the local code.
from contact_audio_perception import spectrogram as S
from contact_audio_perception.contact_model import ContactModel
from contact_audio_perception.encoders import build_encoder_for_model

import audio_io

from paths import BAGS
from paths import MODELS
from paths import OUT
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_S = 1.0
HOP_S = 0.5


def parse_name(name):
    p = name.split('_')
    return '_'.join(p[:-2]), p[-2], p[-1]


def clip_windows(loaded):
    """Yield (t_start_s, (98,128) fbank) using the pipeline's own function."""
    sr = loaded['sample_rate']
    ch = loaded['channels']
    raw = loaded['data'].reshape(-1)          # interleaved int16, as on the wire
    n_win = int(round(WINDOW_S * sr))
    n_hop = int(round(HOP_S * sr))
    frames = loaded['frames']
    for start in range(0, frames - n_win + 1, n_hop):
        buf = raw[start * ch:(start + n_win) * ch]
        fb, gain = S.buffer_to_fbank(buf, ch, sr)
        yield start / sr, fb, gain


def display_spectrogram(loaded):
    """Whole-clip fbank without per-window peak scaling."""
    mono = S.downmix_to_mono(loaded['data'].reshape(-1), loaded['channels'])
    audio = np.asarray(mono, dtype=np.float32) / 32768.0
    audio = S.resample(audio, loaded['sample_rate'], S.ANALYSIS_SAMPLE_RATE)
    return S.compute_fbank(audio, S.KALDI_V1_OPTIONS)


def main():
    # noise_only_* are ambient references, not task takes: they have no
    # action label and must not become a fourth class.
    bags = sorted(p for p in BAGS.iterdir()
                  if p.is_dir() and not p.name.startswith('noise_only'))

    # ---- window features -------------------------------------------------
    feats, meta = [], []
    disp = {}
    for d in bags:
        L = audio_io.load_bag(d)
        action, noise, take = parse_name(d.name)
        n = 0
        for t, fb, gain in clip_windows(L):
            if fb.shape != (98, 128):
                raise AssertionError(f'{d.name} @ {t}s gave {fb.shape}, expected (98,128)')
            feats.append(fb)
            meta.append({'clip': d.name, 'action': action, 'noise': noise,
                         'take': take, 't': t, 'peak_gain': float(gain)})
            n += 1
        disp[d.name] = display_spectrogram(L)
        print(f'{d.name:34s} {n:4d} windows of (98,128)')

    X = np.stack(feats)
    print(f'\nfbank tensor: {X.shape}  (windows, frames, mels)')
    np.save(OUT / 'fbank_windows.npy', X)
    np.savez_compressed(OUT / 'display_spectrograms.npz', **disp)
    import json
    with open(OUT / 'window_meta.json', 'w') as f:
        json.dump(meta, f)

    # ---- embeddings ------------------------------------------------------
    for enc_name in ('cnn', 'melstats', 'ast'):
        bundle = MODELS / f'contact_model_{enc_name}'
        model = ContactModel.load(bundle)
        enc = build_encoder_for_model(model)
        print(f'\n{enc_name}: encoder_id={enc.encoder_id} '
              f'preprocessing={enc.required_preprocessing_id}')
        embs = []
        for i, fb in enumerate(X):
            embs.append(np.asarray(enc.embed(fb), dtype=np.float32).reshape(-1))
            if (i + 1) % 100 == 0:
                print(f'   {i+1}/{len(X)}')
        E = np.stack(embs)
        print(f'   embeddings {E.shape}')
        np.save(OUT / f'emb_{enc_name}.npy', E)

    print(f'\nwrote features and embeddings to {OUT}')


if __name__ == '__main__':
    main()
