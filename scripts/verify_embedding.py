"""
Check /audio/embedding against its source spectrogram, live, on a running graph.

Four claims, each checked against something independent of the node under test:

  IDENTITY     the header is COPIED from the source Spectrogram, not re-stamped.
               Checked bit for bit (sec AND nanosec), paired by
               start_sample_index, which is run-independent. A node that called
               now() would still produce monotonic, plausible stamps -- only an
               exact comparison with the source catches it.
  PROVENANCE   encoder_id, encoder_type, embedding_dim and preprocessing_id are
               populated, and preprocessing_id equals the SOURCE spectrogram's.
               Without that a consumer cannot tell whether two embeddings are
               comparable.
  CONTENT      the vector equals what the same encoder produces from the same
               filterbank, recomputed here from the spectrogram message. This
               is what proves the published embedding is the one that was
               classified, rather than a stale or zeroed buffer.
  RATE         predictions and embeddings arrive at the same rate, and the
               embedding is never LATER than its own prediction -- the ordering
               the node promises by publishing before it classifies.

Usage:
    python3 scripts/verify_embedding.py 30 --model <bundle dir>
"""
import argparse
import os
import sys

from contact_audio_interfaces.msg import AudioEmbedding, ContactPrediction, Spectrogram
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'src',
    'contact_audio_perception'))
from contact_audio_perception.contact_model import ContactModel  # noqa: E402
from contact_audio_perception.encoders import build_encoder_for_model  # noqa: E402

# Recomputation runs the identical code on identical input, so it should be
# bit-exact. A tiny tolerance is allowed only for float32 message round-tripping.
CONTENT_TOLERANCE = 1e-6


class EmbeddingVerifier(Node):
    """Collect spectrograms, embeddings and predictions, and pair them up."""

    def __init__(self):
        """Subscribe to all three topics with the pipeline's own QoS."""
        super().__init__('verify_embedding')
        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.spectrograms = {}
        self.embeddings = {}
        self.predictions = {}
        self.create_subscription(Spectrogram, '/audio/spectrogram', self.on_spec, qos)
        self.create_subscription(AudioEmbedding, '/audio/embedding', self.on_emb, qos)
        self.create_subscription(
            ContactPrediction, '/audio/contact_prediction', self.on_pred, qos)

    def on_spec(self, msg):
        """Record one spectrogram by its sample index."""
        self.spectrograms[int(msg.start_sample_index)] = msg

    def on_emb(self, msg):
        """Record one embedding and the wall time it arrived."""
        self.embeddings[int(msg.start_sample_index)] = (
            msg, self.get_clock().now().nanoseconds)

    def on_pred(self, msg):
        """Record one prediction and the wall time it arrived."""
        self.predictions[int(msg.start_sample_index)] = (
            msg, self.get_clock().now().nanoseconds)


def main():
    """Collect for a while, then check every claim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seconds', type=float, nargs='?', default=30.0)
    parser.add_argument('--model', default='',
                        help='model bundle, to recompute the embedding independently')
    parser.add_argument('--min-samples', type=int, default=5)
    args = parser.parse_args()

    rclpy.init()
    node = EmbeddingVerifier()
    deadline = node.get_clock().now().nanoseconds + int(args.seconds * 1e9)
    try:
        while rclpy.ok() and node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    specs, embs, preds = node.spectrograms, node.embeddings, node.predictions
    node.destroy_node()
    rclpy.shutdown()

    print(f'spectrograms       : {len(specs)}')
    print(f'embeddings         : {len(embs)}')
    print(f'predictions        : {len(preds)}')

    paired = sorted(set(specs) & set(embs))
    print(f'paired by index    : {len(paired)}')
    if len(paired) < args.min_samples:
        print(f'\nRESULT: FAIL (only {len(paired)} paired messages, need '
              f'{args.min_samples}; is the pipeline running?)')
        raise SystemExit(1)

    # ---- identity ----
    stamp_mismatches = 0
    for index in paired:
        spec, (emb, _) = specs[index], embs[index]
        if (spec.header.stamp.sec != emb.header.stamp.sec or
                spec.header.stamp.nanosec != emb.header.stamp.nanosec or
                spec.header.frame_id != emb.header.frame_id):
            stamp_mismatches += 1
    print()
    print('=== identity ===')
    print(f'headers differing from the source : {stamp_mismatches} (must be 0)')

    # ---- provenance ----
    sample = embs[paired[0]][0]
    print()
    print('=== provenance ===')
    print(f'encoder_id         : {sample.encoder_id!r}')
    print(f'encoder_type       : {sample.encoder_type!r}')
    print(f'embedding_dim      : {sample.embedding_dim}')
    print(f'preprocessing_id   : {sample.preprocessing_id!r}')
    provenance_ok = bool(
        sample.encoder_id and sample.encoder_type and sample.preprocessing_id and
        sample.embedding_dim == len(sample.embedding))
    pre_mismatch = sum(
        1 for i in paired if embs[i][0].preprocessing_id != specs[i].preprocessing_id)
    dim_mismatch = sum(
        1 for i in paired if embs[i][0].embedding_dim != len(embs[i][0].embedding))
    print(f'preprocessing_id != source        : {pre_mismatch} (must be 0)')
    print(f'embedding_dim != actual length    : {dim_mismatch} (must be 0)')

    # ---- content ----
    print()
    print('=== content ===')
    content_ok = None
    worst = float('nan')
    if args.model:
        model = ContactModel.load(args.model)
        encoder = build_encoder_for_model(model)
        errors = []
        for index in paired[:20]:
            spec, (emb, _) = specs[index], embs[index]
            fbank = np.asarray(spec.data, dtype=np.float32).reshape(
                int(spec.n_frames), int(spec.n_mels))
            recomputed = encoder.embed(fbank)
            published = np.asarray(emb.embedding, dtype=np.float32)
            if recomputed.shape != published.shape:
                errors.append(float('inf'))
                continue
            errors.append(float(np.abs(recomputed - published).max()))
        worst = max(errors)
        print(f'recomputed from the spectrogram with {encoder.encoder_id}')
        print(f'worst |published - recomputed|    : {worst:.3e} '
              f'(tolerance {CONTENT_TOLERANCE:g})')
        # Sensitivity: the same comparison against a DIFFERENT window must fail,
        # otherwise the encoder output is constant and the check is vacuous.
        control = float(np.abs(
            np.asarray(embs[paired[0]][0].embedding, dtype=np.float32) -
            np.asarray(embs[paired[-1]][0].embedding, dtype=np.float32)).max())
        print(f'control, two different windows    : {control:.3e} '
              f'(must be >> the tolerance, else embeddings are constant)')
        content_ok = worst < CONTENT_TOLERANCE and control > CONTENT_TOLERANCE * 100
    else:
        print('no --model given; content NOT verified')

    # ---- rate and ordering ----
    print()
    print('=== rate and ordering ===')
    both = sorted(set(embs) & set(preds))
    late = sum(1 for i in both if embs[i][1] > preds[i][1])
    ratio = len(embs) / max(len(preds), 1)
    print(f'embeddings / predictions          : {ratio:.3f} (expect 1.000)')
    print(f'embeddings arriving after their own prediction: {late}/{len(both)} '
          f'(must be 0; the node publishes the embedding first)')

    checks = [
        ('header copied unchanged from the source', stamp_mismatches == 0),
        ('provenance fields populated', provenance_ok),
        ('preprocessing_id copied from the spectrogram', pre_mismatch == 0),
        ('embedding_dim matches the vector length', dim_mismatch == 0),
        ('one embedding per prediction', 0.98 <= ratio <= 1.02),
        ('the embedding is never later than its prediction', late == 0),
    ]
    if content_ok is not None:
        checks.insert(4, ('embedding equals an independent recomputation', content_ok))
    print()
    for name, ok in checks:
        print(f'[{"PASS" if ok else "FAIL"}] {name}')
    failed = [n for n, ok in checks if not ok]
    print()
    print('RESULT:', 'EMBEDDING VERIFIED' if not failed
          else f'FAIL ({", ".join(failed)})')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
