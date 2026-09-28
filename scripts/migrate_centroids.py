"""Attach the feature contract to an untagged legacy centroid model.

Re-tagging (rather than retraining) is only valid because gate 2 proves the new
feature path reproduces the legacy embeddings exactly -- the same centroids
therefore still describe the same embedding space. This script refuses to run
unless that is true, so the guarantee cannot be assumed by accident.

Usage: python3 scripts/migrate_centroids.py <legacy.npz> <out.npz> [raw_dir]
"""
import glob
import os
import subprocess
import sys

from contact_audio_perception.contact_model import ContactModel
from contact_audio_perception.encoders.ast_fbank import DEFAULT_MODEL_ID
from contact_audio_perception.identity import dataset_digest
from contact_audio_perception.spectrogram import FbankConfig

USAGE = ('usage: python3 scripts/migrate_centroids.py <legacy.npz> '
         '<out_bundle_dir> [raw_dir]')
WINDOW_SEC = 1.0


def main():
    """Verify parity, then write a tagged copy of the legacy model."""
    if len(sys.argv) < 3:
        print(USAGE, file=sys.stderr)
        raise SystemExit(2)
    legacy_path, out_path = sys.argv[1], sys.argv[2]
    raw_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser('~/train_data')

    here = os.path.dirname(os.path.abspath(__file__))
    print('Re-running the parity gates before migrating...')
    gates = subprocess.run(
        [sys.executable, os.path.join(here, 'verify_block2.py'), raw_dir,
         '--model', legacy_path],
        capture_output=True, text=True)
    if gates.returncode != 0:
        sys.stdout.write(gates.stdout)
        sys.stderr.write(gates.stderr)
        print('\nRefusing to migrate: the parity gates did not pass. The legacy '
              'centroids are only reusable while the embeddings are equivalent.',
              file=sys.stderr)
        raise SystemExit(1)
    print(gates.stdout.strip().splitlines()[-1])

    from contact_audio_perception.encoders.ast_fbank import AstFbankEncoder

    # The promised class -> clip mapping, not a free-text note: it is what makes
    # 'what was this trained on' answerable later.
    wavs = sorted(glob.glob(os.path.join(raw_dir, '*', '*.wav')))
    trained_on = {}
    for wav in wavs:
        label = os.path.basename(os.path.dirname(wav))
        trained_on.setdefault(label, []).append(os.path.basename(wav))
    if not wavs:
        print(f'error: no WAVs under {raw_dir}', file=sys.stderr)
        raise SystemExit(2)

    config = FbankConfig()
    legacy = ContactModel.load(legacy_path, allow_needs_migration=True)
    # Build the encoder so the identity document -- config, checkpoint id AND
    # immutable revision -- comes from the real thing rather than being retyped.
    encoder = AstFbankEncoder(config=config)
    tagged = ContactModel(
        class_names=legacy.class_names,
        centroids=legacy.centroids,
        encoder_type=encoder.encoder_type,
        encoder_config=encoder.encoder_config,
        weights_source='checkpoint',
        checkpoint_id=DEFAULT_MODEL_ID,
        checkpoint_revision=encoder.checkpoint_revision,
        preprocessing_id=config.preprocessing_id,
        expected_n_frames=config.frames_for(WINDOW_SEC),
        expected_n_mels=config.n_mels,
        window_duration_sec=WINDOW_SEC,
        embedding_dim=int(legacy.centroids.shape[1]),
        dataset_digest=dataset_digest(wavs, raw_dir),
        trained_on=trained_on,
        fold_id='all',
    )
    if tagged.encoder_id != encoder.encoder_id:
        print(f'identity mismatch: model {tagged.encoder_id} vs encoder '
              f'{encoder.encoder_id}', file=sys.stderr)
        raise SystemExit(1)
    tagged.save(out_path)
    print(f'wrote bundle {out_path}')
    print(f'  {tagged.describe()}')


if __name__ == '__main__':
    main()
