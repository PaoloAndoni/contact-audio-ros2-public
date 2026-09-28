"""
Train the deployable all-data bundles for the local extractors.

These are the models that ship: fold_id='all', trained on every clip. Their
accuracy is NOT the reported number -- a model fitted on all the data cannot be
scored on it. The honest figures come from evaluate_encoders.py, which retrains
inside every fold.

ast_random is deliberately absent. It is a pretraining control, never a
deployment candidate, and its 345 MB artifact belongs under eval_runs/.

Usage: python3 scripts/train_final_models.py [raw_dir] [--encoders cnn,melstats]
"""
import argparse
import os
from pathlib import Path
import sys

from contact_audio_perception.contact_model import ContactModel
from contact_audio_perception.encoders import build_encoder
from contact_audio_perception.identity import dataset_digest, sha256_hex
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_encoders import centroids_from, load_windows, WINDOW_SEC  # noqa: E402

DEPLOYABLE = ('cnn', 'melstats')


def main():
    """Train and write one bundle per deployable extractor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_dir', nargs='?',
                        default=os.path.expanduser('~/train_data'))
    parser.add_argument('--models', default='src/contact_audio_perception/models')
    parser.add_argument('--encoders', default=','.join(DEPLOYABLE))
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    wanted = [e.strip() for e in args.encoders.split(',') if e.strip()]
    if any(e == 'ast_random' for e in wanted):
        print('ast_random is an evaluation control and is never deployed',
              file=sys.stderr)
        raise SystemExit(2)

    rows, config = load_windows(args.raw_dir)
    class_names = sorted({r['label'] for r in rows})
    labels = [r['label'] for r in rows]
    wavs = sorted(str(p) for p in Path(args.raw_dir).glob('*/*.wav'))
    digest = dataset_digest(wavs, args.raw_dir)
    trained_on = {}
    for wav in wavs:
        trained_on.setdefault(os.path.basename(os.path.dirname(wav)), []).append(
            os.path.basename(wav))
    print(f'{len(rows)} windows | classes {class_names} | dataset {digest}')

    for name in wanted:
        if name == 'cnn':
            encoder = build_encoder('cnn', n_classes=len(class_names), config=config)
            loss = encoder.fit([r['fbank'] for r in rows],
                               [class_names.index(lab) for lab in labels],
                               epochs=args.epochs)
            payload = encoder.state_bytes()
            encoder.weights_sha256 = sha256_hex(payload)
            weights = {'weights_source': 'bundled',
                       'weights_file': 'encoder.safetensors',
                       'weights_sha256': encoder.weights_sha256,
                       'weights_format': 'safetensors'}
            print(f'cnn: trained on all {len(rows)} windows, final loss {loss:.4f}')
        else:
            encoder = build_encoder(name, config=config)
            payload = None
            weights = {'weights_source': 'none'}

        embeddings = [encoder.embed(r['fbank']) for r in rows]
        cents = centroids_from(embeddings, labels, class_names)
        bundle = Path(args.models) / f'contact_model_{name}'
        model = ContactModel(
            class_names=class_names, centroids=cents, encoder_type=name,
            encoder_config=encoder.encoder_config,
            preprocessing_id=config.preprocessing_id,
            expected_n_frames=int(rows[0]['fbank'].shape[0]),
            expected_n_mels=int(rows[0]['fbank'].shape[1]),
            window_duration_sec=WINDOW_SEC, fold_id='all',
            trained_on=trained_on, dataset_digest=digest,
            embedding_dim=int(np.asarray(embeddings[0]).size), **weights)
        if model.encoder_id != encoder.encoder_id:
            print(f'identity mismatch for {name}: {model.encoder_id} != '
                  f'{encoder.encoder_id}', file=sys.stderr)
            raise SystemExit(1)
        model.save(bundle, weights_bytes=payload)
        print(f'  wrote {bundle}')
        print(f'    {model.describe()}')


if __name__ == '__main__':
    main()
