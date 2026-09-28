"""Leave-one-trial-out cross-validation for the contact classifier.

Encodes every window of every class-folder WAV once, then for each held-out
trial index trains centroids on the remaining trials and classifies the held-out
clips. Reports per-window and per-clip (majority-vote) accuracy plus a confusion
matrix (printed and saved as a PNG). This is an honest generalization estimate:
no test window ever contributes to the centroids it is scored against.

Usage: python3 scripts/evaluate.py <raw_dir> [out_png]
"""
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

USAGE = 'usage: python3 scripts/evaluate.py <raw_dir> [out_png]'


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


def print_confusion(cm, row_labels, col_labels):
    width = max(len(x) for x in row_labels + col_labels) + 2
    print('true\\pred'.ljust(width) + ''.join(f'{c:>{width}}' for c in col_labels))
    for i, row in enumerate(row_labels):
        cells = ''.join(f'{cm[i, j]:>{width}}' for j in range(len(col_labels)))
        print(row.ljust(width) + cells)


def save_confusion_png(cm, row_labels, col_labels, path, clip_acc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > vmax / 2 else 'black')
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(f'Leave-one-trial-out CV  |  per-clip accuracy = {clip_acc:.2f}')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print('saved', path)


def main():
    if len(sys.argv) < 2:
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    raw_dir = Path(sys.argv[1])
    if not raw_dir.is_dir():
        print(f'error: not a directory: {raw_dir}\n{USAGE}', file=sys.stderr)
        raise SystemExit(2)
    out_png = sys.argv[2] if len(sys.argv) > 2 else 'confusion_matrix.png'

    from contact_audio_perception.ast_encoder import AstEncoder, window_to_model_input
    from contact_audio_perception.contact_classifier import CentroidClassifier

    encoder = AstEncoder()
    classes = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())

    # Encode every window of every clip once: data[cls][trial] = (n_win, dim).
    data = defaultdict(dict)
    for cls in classes:
        for wav in sorted((raw_dir / cls).glob('*.wav')):
            trial = wav.stem.split('_')[-1]
            samples, ch, sr = load_int16(wav)
            embs = [encoder.encode(window_to_model_input(w, c, s))
                    for w, c, s in iter_windows(samples, ch, sr)]
            if embs:
                data[cls][trial] = np.array(embs, dtype=np.float32)
    trials = sorted({t for cls in classes for t in data[cls]})
    print(f'classes = {classes}')
    print(f'trials  = {trials} (leave-one-trial-out)')

    win_true, win_pred, clip_true, clip_pred = [], [], [], []
    for held in trials:
        names, cents = [], []
        for cls in classes:
            train = [data[cls][t] for t in data[cls] if t != held]
            if not train:
                continue
            names.append(cls)
            cents.append(np.concatenate(train, axis=0).mean(axis=0))
        clf = CentroidClassifier(names, np.array(cents, dtype=np.float32))
        for cls in classes:
            if held not in data[cls]:
                continue
            preds = [clf.predict(e)[0] for e in data[cls][held]]
            win_true += [cls] * len(preds)
            win_pred += preds
            clip_true.append(cls)
            clip_pred.append(Counter(preds).most_common(1)[0][0])

    win_hits = sum(t == p for t, p in zip(win_true, win_pred))
    clip_hits = sum(t == p for t, p in zip(clip_true, clip_pred))
    clip_acc = clip_hits / len(clip_true)
    print()
    print(f'per-window accuracy: {win_hits / len(win_true):.3f} ({win_hits}/{len(win_true)})')
    print(f'per-clip  accuracy: {clip_acc:.3f} ({clip_hits}/{len(clip_true)})')
    print()

    col_labels = classes + sorted(set(clip_pred) - set(classes))
    col_idx = {lbl: j for j, lbl in enumerate(col_labels)}
    cm = np.zeros((len(classes), len(col_labels)), dtype=int)
    for t, p in zip(clip_true, clip_pred):
        cm[classes.index(t), col_idx[p]] += 1
    print('per-clip confusion matrix (rows=true, cols=pred):')
    print_confusion(cm, classes, col_labels)
    save_confusion_png(cm, classes, col_labels, out_png, clip_acc)


if __name__ == '__main__':
    main()
