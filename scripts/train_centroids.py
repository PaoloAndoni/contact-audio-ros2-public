"""Train a nearest-centroid contact model from class-folder WAVs.

Windows each WAV exactly like audio_window_node (1 s, 0.5 s hop), preprocesses
with the same window_to_model_input() the live node uses, encodes each window
with AstEncoder, and saves per-class mean centroids to an .npz consumable by
contact_classifier_node. Run after building + sourcing the workspace.

Usage:
    python3 scripts/train_centroids.py <raw_dir> [out.npz]

<raw_dir> holds one subfolder per class, each containing *.wav files (16-bit PCM).
"""
import sys
import wave
from pathlib import Path

import numpy as np

USAGE = 'usage: python3 train_centroids.py <raw_dir> [out.npz]'


def load_int16(path):
    with wave.open(str(path), 'rb') as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16), channels, sample_rate


def iter_windows(samples, channels, sample_rate, window_sec=1.0, hop_sec=0.5):
    n_frames = samples.size // channels
    win = int(round(window_sec * sample_rate))
    hop = int(round(hop_sec * sample_rate))
    if channels < 1 or win < 1 or hop < 1 or n_frames < win:
        return
    frames = samples.reshape(-1, channels)
    start = 0
    while start + win <= n_frames:
        yield frames[start:start + win].reshape(-1), channels, sample_rate
        start += hop


def main():
    # Validate the CLI before the heavy torch/transformers import so bad args
    # fail instantly (and with the correct exit code) instead of after a model load.
    if len(sys.argv) < 2:
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    raw_dir = Path(sys.argv[1])
    if not raw_dir.is_dir():
        print(f'error: not a directory: {raw_dir}\n{USAGE}', file=sys.stderr)
        raise SystemExit(2)
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'contact_centroids.npz'

    from contact_audio_perception.ast_encoder import AstEncoder, window_to_model_input

    encoder = AstEncoder()
    class_names = []
    centroids = []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        embeddings = []
        for wav in sorted(class_dir.glob('*.wav')):
            samples, channels, sample_rate = load_int16(wav)
            for window, ch, sr in iter_windows(samples, channels, sample_rate):
                embeddings.append(encoder.encode(window_to_model_input(window, ch, sr)))
        if not embeddings:
            print(f'{class_dir.name}: no usable windows, skipping')
            continue
        class_names.append(class_dir.name)
        centroids.append(np.mean(embeddings, axis=0))
        print(f'{class_dir.name}: {len(embeddings)} windows')

    if len(class_names) < 2:
        print(
            f'error: need >= 2 classes with usable windows, found '
            f'{len(class_names)}: {class_names}',
            file=sys.stderr,
        )
        raise SystemExit(1)

    np.savez(
        out_path,
        class_names=np.array(class_names),
        centroids=np.array(centroids, dtype=np.float32),
    )
    print('saved', out_path, '| classes:', class_names,
          '| shape:', np.array(centroids).shape)


if __name__ == '__main__':
    main()
