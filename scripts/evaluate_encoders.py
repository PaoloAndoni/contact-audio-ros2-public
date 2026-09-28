"""
Leave-one-trial-out comparison of every feature extractor on identical input.

The filterbank is computed ONCE per window and shared by all extractors, which
is the whole point of publishing it: nobody gets to preprocess their input
differently, so accuracy differences are attributable to the extractor.

FOLD ISOLATION. Extractors split into two kinds:
  * FIXED (ast, ast_random, melstats) do not learn from this dataset, so every
    window may be encoded once and then split per fold. The held-out trial never
    influences the encoder because the encoder cannot be influenced.
  * LEARNED (cnn) must be retrained inside every fold, on that fold's training
    clips only. Training once on all windows and then cross-validating the
    centroids would leak the held-out trial into the encoder and make the
    reported accuracy meaningless.
Centroids are always built from the fold's training clips alone.

Per-clip accuracy is the number to trust. The 108 windows come from 12
recordings at 50% overlap, so they are not 108 independent samples.

Usage: python3 scripts/evaluate_encoders.py [raw_dir] [--out eval_runs/<name>]
                                            [--encoders ast,cnn,melstats]
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from contact_audio_perception.contact_classifier import CentroidClassifier
from contact_audio_perception.contact_model import ContactModel
from contact_audio_perception.encoders import build_encoder
from contact_audio_perception.identity import dataset_digest, sha256_hex
from contact_audio_perception.spectrogram import (
    buffer_to_fbank,
    FbankConfig,
    read_wav_int16,
)
import numpy as np

WINDOW_SEC = 1.0
HOP_SEC = 0.5
FIXED_ENCODERS = ('ast', 'ast_random', 'melstats')
LEARNED_ENCODERS = ('cnn',)
ALL_ENCODERS = ('ast', 'ast_random', 'cnn', 'melstats')


def load_windows(raw_dir):
    """Compute every window's filterbank once, tagged with its class and trial."""
    config = FbankConfig()
    rows = []
    for class_dir in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir()):
        for wav in sorted(class_dir.glob('*.wav')):
            trial = wav.stem.split('_')[-1]
            samples, channels, rate = read_wav_int16(wav)
            frames = samples.size // channels
            win = int(round(WINDOW_SEC * rate))
            hop = int(round(HOP_SEC * rate))
            start = 0
            while start + win <= frames:
                block = samples[start * channels:(start + win) * channels]
                fbank, _ = buffer_to_fbank(block, channels, rate, config)
                rows.append({'label': class_dir.name, 'trial': trial,
                             'clip': wav.stem, 'fbank': fbank})
                start += hop
    return rows, config


def centroids_from(embeddings, labels, class_names):
    """Mean embedding per class, in class_names order."""
    return np.stack([
        np.mean([e for e, lab in zip(embeddings, labels) if lab == name], axis=0)
        for name in class_names]).astype(np.float32)


def score(classifier, embeddings, rows):
    """Return (per-window hits, window count, per-clip hits, clip count)."""
    per_clip = {}
    win_hits = 0
    for emb, row in zip(embeddings, rows):
        label = classifier.predict(emb)[0]
        win_hits += int(label == row['label'])
        per_clip.setdefault((row['clip'], row['label']), []).append(label)
    clip_hits = 0
    for (_, truth), labels in per_clip.items():
        majority = max(set(labels), key=labels.count)
        clip_hits += int(majority == truth)
    return win_hits, len(rows), clip_hits, len(per_clip)


def measure_latency(encoder, fbank, repeats=5):
    """Return the median milliseconds one embedding takes."""
    encoder.embed(fbank)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        encoder.embed(fbank)
        times.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(times))


def encoder_size(encoder):
    """Return (parameter count, serialised weight bytes)."""
    model = getattr(encoder, 'model', None)
    if model is None:
        return 0, 0
    params = sum(p.numel() for p in model.parameters())
    payload = encoder.state_bytes() if hasattr(encoder, 'state_bytes') else b''
    return params, len(payload)


def main():
    """Run the comparison and write results plus fold artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_dir', nargs='?',
                        default=os.path.expanduser('~/train_data'))
    parser.add_argument('--out', default='eval_runs/latest')
    parser.add_argument('--encoders', default=','.join(ALL_ENCODERS))
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    wanted = [e.strip() for e in args.encoders.split(',') if e.strip()]
    unknown = [e for e in wanted if e not in ALL_ENCODERS]
    if unknown:
        print(f'unknown encoders: {unknown}', file=sys.stderr)
        raise SystemExit(2)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows, config = load_windows(args.raw_dir)
    if not rows:
        print(f'no windows under {args.raw_dir}', file=sys.stderr)
        raise SystemExit(2)
    class_names = sorted({r['label'] for r in rows})
    trials = sorted({r['trial'] for r in rows})
    wavs = sorted(str(p) for p in Path(args.raw_dir).glob('*/*.wav'))
    digest = dataset_digest(wavs, args.raw_dir)
    print(f'{len(rows)} windows | classes {class_names} | trials {trials} '
          f'(leave-one-trial-out) | dataset {digest}')
    print(f'features {config.preprocessing_id}, {rows[0]["fbank"].shape} per window')
    print()

    # The random-AST artifact is created ONCE per run and shared by every fold:
    # it is fixed, not fold-trained, so re-creating it per fold would only waste
    # 345 MB and time. It stays under eval_runs/ and is never installed.
    shared = {}
    if 'ast_random' in wanted:
        print('creating the shared random-AST control artifact...')
        encoder = build_encoder('ast_random', config=config)
        payload = encoder.state_bytes()
        artifact = out_root / 'ast_random'
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / 'encoder.safetensors').write_bytes(payload)
        shared['ast_random'] = {'sha256': sha256_hex(payload),
                                'path': str(artifact / 'encoder.safetensors')}
        encoder.weights_sha256 = shared['ast_random']['sha256']
        shared['ast_random']['encoder'] = encoder
        print(f'  {len(payload) / 1e6:.1f} MB, sha {shared["ast_random"]["sha256"][:16]}, '
              f'{encoder.encoder_id}')

    results = {}
    for name in wanted:
        print(f'--- {name} ---')
        started = time.perf_counter()
        if name in FIXED_ENCODERS:
            if name == 'ast_random':
                encoder = shared['ast_random']['encoder']
            else:
                encoder = build_encoder(name, config=config)
            print(f'  {encoder.encoder_id}, encoding {len(rows)} windows once...')
            embeddings = [encoder.embed(r['fbank']) for r in rows]
            latency = measure_latency(encoder, rows[0]['fbank'])
            params, weight_bytes = encoder_size(encoder)
            dim = int(len(embeddings[0]))
        else:
            latency = params = weight_bytes = dim = None

        win_hits = win_total = clip_hits = clip_total = 0
        for held in trials:
            train_idx = [i for i, r in enumerate(rows) if r['trial'] != held]
            test_idx = [i for i, r in enumerate(rows) if r['trial'] == held]
            if not test_idx:
                continue

            if name in LEARNED_ENCODERS:
                # Retrained per fold, on this fold's training windows ONLY.
                encoder = build_encoder(
                    name, n_classes=len(class_names), config=config)
                encoder.fit([rows[i]['fbank'] for i in train_idx],
                            [class_names.index(rows[i]['label']) for i in train_idx],
                            epochs=args.epochs)
                fold_train = [encoder.embed(rows[i]['fbank']) for i in train_idx]
                fold_test = [encoder.embed(rows[i]['fbank']) for i in test_idx]
                if latency is None:
                    latency = measure_latency(encoder, rows[0]['fbank'])
                    params, weight_bytes = encoder_size(encoder)
                    dim = int(len(fold_train[0]))
                payload = encoder.state_bytes()
                encoder.weights_sha256 = sha256_hex(payload)
            else:
                fold_train = [embeddings[i] for i in train_idx]
                fold_test = [embeddings[i] for i in test_idx]

            cents = centroids_from(fold_train,
                                   [rows[i]['label'] for i in train_idx], class_names)
            classifier = CentroidClassifier(class_names, cents)
            w_hit, w_tot, c_hit, c_tot = score(
                classifier, fold_test, [rows[i] for i in test_idx])
            win_hits += w_hit
            win_total += w_tot
            clip_hits += c_hit
            clip_total += c_tot

            if name in LEARNED_ENCODERS:
                # Fold artifacts are evaluation outputs. save() refuses to place
                # them under any models/ tree even with allow_fold=True.
                bundle = out_root / name / f'fold_{held}'
                ContactModel(
                    class_names=class_names, centroids=cents,
                    encoder_type=name, encoder_config=encoder.encoder_config,
                    weights_source='bundled', weights_file='encoder.safetensors',
                    weights_sha256=encoder.weights_sha256,
                    weights_format='safetensors',
                    preprocessing_id=config.preprocessing_id,
                    expected_n_frames=int(rows[0]['fbank'].shape[0]),
                    expected_n_mels=int(rows[0]['fbank'].shape[1]),
                    window_duration_sec=WINDOW_SEC, fold_id=held,
                    dataset_digest=digest,
                ).save(bundle, weights_bytes=payload, allow_fold=True)

        results[name] = {
            'encoder_id': encoder.encoder_id,
            'embedding_dim': dim,
            'per_window': win_hits / win_total if win_total else 0.0,
            'per_window_counts': [win_hits, win_total],
            'per_clip': clip_hits / clip_total if clip_total else 0.0,
            'per_clip_counts': [clip_hits, clip_total],
            'latency_ms': latency,
            'parameters': params,
            'weight_bytes': weight_bytes,
            'trained_per_fold': name in LEARNED_ENCODERS,
            'elapsed_sec': round(time.perf_counter() - started, 1),
        }
        r = results[name]
        print(f'  per-clip {r["per_clip"]:.3f} ({clip_hits}/{clip_total}) | '
              f'per-window {r["per_window"]:.3f} ({win_hits}/{win_total}) | '
              f'{latency:.1f} ms/window | {params} params')
        print()

    print(f'{"extractor":<12}{"role":<24}{"clip":>8}{"window":>9}'
          f'{"ms/win":>9}{"params":>12}{"dim":>6}')
    print('-' * 80)
    roles = {'ast': 'pretrained neural', 'ast_random': 'pretraining control',
             'cnn': 'learned neural', 'melstats': 'non-neural baseline'}
    for name in wanted:
        r = results[name]
        print(f'{name:<12}{roles[name]:<24}{r["per_clip"]:>8.3f}'
              f'{r["per_window"]:>9.3f}{r["latency_ms"]:>9.1f}'
              f'{r["parameters"]:>12,}{r["embedding_dim"]:>6}')

    payload = {'dataset_digest': digest, 'classes': class_names, 'trials': trials,
               'windows': len(rows), 'preprocessing_id': config.preprocessing_id,
               'results': results}
    (out_root / 'results.json').write_text(json.dumps(payload, indent=2))
    print()
    print(f'wrote {out_root / "results.json"}')


if __name__ == '__main__':
    main()
