"""Staged parity gate: prove the new feature path reproduces the legacy one.

Gate 1  input tensor   new prepare_input(fbank)  vs  ASTFeatureExtractor
Gate 2  embedding      new encoder CLS           vs  legacy CLS  (+ determinism)
Gate 3  classification both embeddings through the bundled centroid model

Nothing downstream of this -- retraining, retagging the model, retiring the
legacy nodes -- may proceed until all three gates pass.

Usage: python3 scripts/verify_block2.py [raw_dir] [--atol 1e-5]
"""
import argparse
import glob
import os
import sys

from contact_audio_perception.contact_model import ContactModel
from contact_audio_perception.encoders.ast_fbank import AstFbankEncoder
from contact_audio_perception.spectrogram import (
    buffer_to_fbank,
    downmix_to_mono,
    FbankConfig,
    read_wav_int16,
)
import numpy as np

DEFAULT_RAW = os.path.expanduser('~/train_data')
DEFAULT_MODEL = os.path.expanduser(
    '~/audio-manipulation/ros2_ws/src/contact_audio_perception/models/'
    'contact_centroids.npz')


def one_second_windows(raw_dir, per_class=2, duration=1.0):
    """Yield (label, samples, channels, rate) for a few windows per class."""
    for class_dir in sorted(p for p in glob.glob(os.path.join(raw_dir, '*'))
                            if os.path.isdir(p)):
        label = os.path.basename(class_dir)
        for wav in sorted(glob.glob(os.path.join(class_dir, '*.wav')))[:per_class]:
            samples, channels, rate = read_wav_int16(wav)
            width = int(duration * rate) * channels
            # Take a window from the middle, where the actual contact event is.
            start = ((samples.size // 2) // channels) * channels
            window = samples[start:start + width]
            if window.size == width:
                yield label, window, channels, rate


def legacy_tensor_and_embedding(samples, channels, rate, extractor, model):
    """Run the original four-node preprocessing and AST forward pass."""
    import torch
    from contact_audio_perception.ast_encoder import window_to_model_input

    mono = downmix_to_mono(samples, channels, legacy_int16=True)
    audio = window_to_model_input(mono, 1, rate)
    tensor = extractor(audio, sampling_rate=16000, return_tensors='pt')['input_values']
    with torch.no_grad():
        cls = model(input_values=tensor).last_hidden_state[:, 0, :]
    return tensor.numpy()[0], cls.squeeze(0).numpy().astype(np.float32)


def main():
    """Run all three gates and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_dir', nargs='?', default=DEFAULT_RAW)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    args = parser.parse_args()

    from transformers import ASTFeatureExtractor

    config = FbankConfig()
    print(f'feature config : {config.describe()}')
    encoder = AstFbankEncoder(config=config)
    extractor = ASTFeatureExtractor.from_pretrained(encoder.model_id)
    print(f'encoder        : {encoder.encoder_id} dim={encoder.embedding_dim} '
          f'max_length={encoder.max_length} mean={encoder.mean} std={encoder.std}')

    model = ContactModel.load(args.model, allow_needs_migration=True, allow_fold=True)
    classifier = model.classifier()
    print(f'model          : {model.describe()}')
    print()

    windows = list(one_second_windows(args.raw_dir))
    if not windows:
        print(f'no windows found under {args.raw_dir}', file=sys.stderr)
        raise SystemExit(2)

    g1 = g2 = g3 = True
    worst_tensor = worst_emb = 0.0
    worst_cos = 1.0
    worst_prob = 0.0
    print(f'{"window":<22}{"tensor max|d|":>15}{"emb max|d|":>13}'
          f'{"cosine":>11}{"labels":>18}')
    print('-' * 79)

    for label, samples, channels, rate in windows:
        fbank, _ = buffer_to_fbank(samples, channels, rate, config)
        mine_tensor = encoder.prepare_input(fbank).numpy()[0]
        mine_emb = encoder.embed(fbank)
        legacy_tensor, legacy_emb = legacy_tensor_and_embedding(
            samples, channels, rate, extractor, model=encoder.model)

        # Gate 1
        d_tensor = float(np.max(np.abs(mine_tensor - legacy_tensor)))
        ok1 = (mine_tensor.shape == (encoder.max_length, encoder.n_mels)
               and mine_tensor.shape == legacy_tensor.shape
               and np.allclose(mine_tensor, legacy_tensor, atol=args.atol, rtol=args.rtol))

        # Gate 2
        d_emb = float(np.max(np.abs(mine_emb - legacy_emb)))
        cos = float(np.dot(mine_emb, legacy_emb)
                    / (np.linalg.norm(mine_emb) * np.linalg.norm(legacy_emb) + 1e-12))
        repeat = encoder.embed(fbank)
        ok2 = (mine_emb.shape == (encoder.embedding_dim,)
               and np.allclose(mine_emb, legacy_emb, atol=args.atol, rtol=args.rtol)
               and np.array_equal(mine_emb, repeat))

        # Gate 3
        mine_pred = classifier.predict(mine_emb)
        legacy_pred = classifier.predict(legacy_emb)
        d_prob = float(np.max(np.abs(np.array(mine_pred[3]) - np.array(legacy_pred[3]))))
        ok3 = (mine_pred[0] == legacy_pred[0]
               and mine_pred[2] == legacy_pred[2]
               and abs(mine_pred[1] - legacy_pred[1]) < 1e-4
               and d_prob < 1e-4)

        g1 &= ok1
        g2 &= ok2
        g3 &= ok3
        worst_tensor = max(worst_tensor, d_tensor)
        worst_emb = max(worst_emb, d_emb)
        worst_cos = min(worst_cos, cos)
        worst_prob = max(worst_prob, d_prob)
        marks = ''.join('.' if ok else 'X' for ok in (ok1, ok2, ok3))
        print(f'{label + " " + str(rate):<22}{d_tensor:>15.3e}{d_emb:>13.3e}'
              f'{cos:>11.8f}{mine_pred[0] + "/" + legacy_pred[0]:>16} {marks}')

    print('-' * 79)
    print(f'windows tested        : {len(windows)}')
    print(f'GATE 1 input tensor   : {"PASS" if g1 else "FAIL"} '
          f'(worst |d| {worst_tensor:.3e}, tol {args.atol:g})')
    print(f'GATE 2 embedding      : {"PASS" if g2 else "FAIL"} '
          f'(worst |d| {worst_emb:.3e}, worst cosine {worst_cos:.10f}, deterministic)')
    print(f'GATE 3 classification : {"PASS" if g3 else "FAIL"} '
          f'(worst prob |d| {worst_prob:.3e})')
    ok = g1 and g2 and g3
    print()
    print('RESULT:', 'ALL GATES PASS' if ok else 'FAILED')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
