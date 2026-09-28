#!/usr/bin/env python3
"""
contact_ml_node - spectrogram in, contact prediction out.

Replaces ast_encoder_node + contact_classifier_node:

    /audio/spectrogram -> validate -> compatibility check -> encoder
                       -> /audio/embedding
                       -> CentroidClassifier -> /audio/contact_prediction

TWO OUTPUTS, ONE FORWARD PASS: the embedding is published as well as the label.
A policy needs the features, and an argmax over three classes throws away all
but log2(3) bits of a 128- or 768-D vector. The embedding is published BEFORE
classification, so nothing downstream of it can delay it, and the publish cost
is measured and reported next to the inference time rather than assumed free.

RATE, HONESTLY: the DAQ node publishes at 15 Hz, but AST takes 1.6-1.8 s per
window on this CPU when both nodes are running (~0.6 Hz achieved; it measures
1.15 s in isolation, so quote the live figure, not that one). This node does NOT
classify at 15 Hz. It subscribes best-effort with depth 1 so that while one
inference runs, only the newest spectrogram survives and everything older is
discarded. Predictions therefore carry roughly one inference period of latency,
which the node reports rather than hides.

The header (capture time of the buffer's first sample) is propagated untouched,
so a prediction still points at when the sound happened, not when it was
classified.
"""
import time

from contact_audio_interfaces.msg import AudioEmbedding, ContactPrediction, Spectrogram
from contact_audio_perception.contact_model import ContactModel
from contact_audio_perception.encoders import build_encoder, build_encoder_for_model
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# Out of range for both settings it guards, so it cannot collide with a real one.
USE_MODEL_VALUE = -999.0


class ContactMlNode(Node):
    """Classify incoming spectrograms into contact predictions."""

    def __init__(self):
        """Load the model and encoder, then subscribe."""
        super().__init__('contact_ml_node')

        self.declare_parameter('input_topic', '/audio/spectrogram')
        self.declare_parameter('output_topic', '/audio/contact_prediction')
        self.declare_parameter('embedding_topic', '/audio/embedding')
        self.declare_parameter('publish_embedding', True)
        self.declare_parameter('model_path', '')
        self.declare_parameter('encoder_type', '')
        self.declare_parameter('unknown_threshold', USE_MODEL_VALUE)
        self.declare_parameter('temperature', USE_MODEL_VALUE)
        self.declare_parameter('allow_legacy_model', False)
        self.declare_parameter('stats_period_sec', 10.0)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        model_path = self.get_parameter('model_path').value
        allow_legacy = bool(self.get_parameter('allow_legacy_model').value)

        if not model_path:
            self.get_logger().error(
                "Parameter 'model_path' is required (a model bundle directory "
                'from scripts/migrate_centroids.py or train_centroids.py).')
            raise SystemExit(1)
        try:
            self.model = ContactModel.load(
                model_path, allow_needs_migration=allow_legacy)
        except Exception as exc:
            self.get_logger().error(f'Failed to load model {model_path!r}: {exc}')
            raise SystemExit(1) from exc
        if self.model.needs_migration:
            self.get_logger().warn(
                'Running on a PRE-V3 model: incoming spectrograms cannot be checked '
                'against the features or encoder it was trained on. For diagnosis '
                'only -- migrate it before trusting any result.')

        # Build from the bundle, so the pinned checkpoint revision decides which
        # weights load. Only an explicit encoder_type override goes the other way,
        # and the identity check below still has to pass.
        override = self.get_parameter('encoder_type').value
        try:
            if override and override != self.model.encoder_type:
                self.encoder = build_encoder(override)
            else:
                self.encoder = build_encoder_for_model(self.model)
        except Exception as exc:
            self.get_logger().error(f'Failed to build the encoder: {exc}')
            raise SystemExit(1) from exc
        mismatch = self.model.encoder_mismatch(
            self.encoder.encoder_type, self.encoder.encoder_id,
            self.encoder.embedding_dim)
        if mismatch is not None:
            self.get_logger().error(f'Encoder does not fit the model: {mismatch}')
            raise SystemExit(1)

        # Deployment defaults travel with the centroids, because different
        # encoders produce different similarity distributions. USE_MODEL_VALUE is
        # a sentinel rather than a plain default: unknown_threshold legitimately
        # ranges over [-1, 1] and temperature over (0, inf), so no in-range value
        # can mean "unset". An explicit override still wins, and is logged.
        overrides = {}
        for name in ('unknown_threshold', 'temperature'):
            value = float(self.get_parameter(name).value)
            if value != USE_MODEL_VALUE:
                overrides[name] = value
        if overrides:
            self.get_logger().warn(
                f'Overriding classifier settings stored in the model '
                f'({self.model.classifier_config}) with {overrides}.')
        self.classifier = self.model.classifier(
            overrides.get('unknown_threshold'), overrides.get('temperature'))

        # Depth 1, best effort: while an inference runs, keep only the newest
        # spectrogram. Queueing them would build an ever-growing backlog of
        # predictions about audio that is already seconds old.
        sub_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        pub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        self.publisher = self.create_publisher(ContactPrediction, output_topic, pub_qos)

        # Embeddings go out best-effort with depth 1, the mirror image of the
        # input: a policy wants the freshest features, and a reliable queue
        # would make this node block on a slow consumer and so change the rate
        # it reports for the classification path.
        embedding_topic = self.get_parameter('embedding_topic').value
        self.embedding_publisher = None
        if bool(self.get_parameter('publish_embedding').value) and embedding_topic:
            emb_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST)
            self.embedding_publisher = self.create_publisher(
                AudioEmbedding, embedding_topic, emb_qos)

        self.subscription = self.create_subscription(
            Spectrogram, input_topic, self.on_spectrogram, sub_qos)

        self._processed = 0
        self._rejected = 0
        self._latencies = []
        self._inference_ms = []
        self._embed_publish_us = []
        self._logged_first = False
        stats_period = float(self.get_parameter('stats_period_sec').value)
        if stats_period > 0.0:
            self.create_timer(stats_period, self._log_stats)

        self.get_logger().info(f'Model   : {self.model.describe()}')
        self.get_logger().info(
            f'Encoder : {self.encoder.encoder_id} dim={self.encoder.embedding_dim}')
        self.get_logger().info(f"Classifying '{input_topic}' -> '{output_topic}'")
        if self.embedding_publisher is not None:
            self.get_logger().info(f"Embeddings -> '{embedding_topic}'")
        else:
            self.get_logger().warn(
                'publish_embedding is off: the encoder output is discarded after '
                'the argmax, so nothing downstream can learn a policy from it.')

    def _reject(self, reason):
        """Count and log a rejected message without spamming the console."""
        self._rejected += 1
        self.get_logger().warn(f'Dropping spectrogram: {reason}',
                               throttle_duration_sec=5.0)

    def _publish_embedding(self, msg, embedding):
        """
        Publish one embedding and return what that cost, in microseconds.

        Returns None when the publisher is disabled, so the caller can tell
        "not published" from "published in ~0 us".
        """
        if self.embedding_publisher is None:
            return None
        started = time.perf_counter()
        out = AudioEmbedding()
        out.header = msg.header          # capture time, copied, never re-stamped
        out.start_sample_index = int(msg.start_sample_index)
        out.encoder_id = self.encoder.encoder_id
        out.encoder_type = self.encoder.encoder_type
        out.embedding_dim = int(embedding.shape[0])
        # From the SPECTROGRAM, not from the model: it states which features were
        # actually encoded. The model's own id is already checked against it above.
        out.preprocessing_id = msg.preprocessing_id
        out.embedding = embedding.astype(np.float32).tolist()
        self.embedding_publisher.publish(out)
        return (time.perf_counter() - started) * 1e6

    def on_spectrogram(self, msg):
        """Encode and classify one spectrogram, then publish the prediction."""
        expected = int(msg.n_frames) * int(msg.n_mels)
        if msg.n_frames < 1 or msg.n_mels < 1 or len(msg.data) != expected:
            self._reject(f'data length {len(msg.data)} does not match '
                         f'{msg.n_frames}x{msg.n_mels}')
            return

        # Pass the id through as-is: '' must be rejected as missing, not treated
        # as "nothing to check".
        reason = self.model.incompatibility(
            preprocessing_id=msg.preprocessing_id,
            n_frames=int(msg.n_frames), n_mels=int(msg.n_mels),
            encoder_type=self.encoder.encoder_type)
        if reason is not None and not self.model.legacy:
            self._reject(reason)
            return

        fbank = np.asarray(msg.data, dtype=np.float32).reshape(
            int(msg.n_frames), int(msg.n_mels))
        started = time.perf_counter()
        try:
            embedding = self.encoder.embed(fbank)
        except ValueError as exc:
            self._reject(str(exc))
            return
        inference_ms = (time.perf_counter() - started) * 1000.0

        # Straight after the forward pass, before the classifier touches it.
        publish_us = self._publish_embedding(msg, embedding)

        label, confidence, class_names, probs = self.classifier.predict(embedding)

        out = ContactPrediction()
        out.header = msg.header          # capture time, never re-stamped
        out.start_sample_index = int(msg.start_sample_index)
        out.label = label
        out.confidence = confidence
        out.class_names = class_names
        out.probabilities = probs
        self.publisher.publish(out)

        # Age of the audio at the moment the prediction goes out: buffer length
        # plus inference, i.e. what a downstream consumer actually has to live with.
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        age_ms = (self.get_clock().now() - stamp).nanoseconds / 1e6
        self._processed += 1
        self._inference_ms.append(inference_ms)
        self._latencies.append(age_ms)
        if publish_us is not None:
            self._embed_publish_us.append(publish_us)

        if not self._logged_first:
            self._logged_first = True
            self.get_logger().info(
                f'First prediction: {label} ({confidence:.2f}) | '
                f'inference {inference_ms:.0f} ms | audio age at publish '
                f'{age_ms:.0f} ms')
        self.get_logger().info(f'{label} ({confidence:.2f})',
                               throttle_duration_sec=1.0)

    def _log_stats(self):
        """Report the rate actually achieved, rather than the rate subscribed to."""
        if not self._inference_ms:
            return
        inf = np.array(self._inference_ms)
        lat = np.array(self._latencies)
        # Report the embedding publish cost in the same breath as the rate it is
        # accused of slowing down, so the claim stays checkable at a glance.
        extra = ''
        if self._embed_publish_us:
            emb = np.array(self._embed_publish_us)
            extra = (f' | embedding publish median {np.median(emb):.0f} us '
                     f'({100.0 * np.median(emb) / 1000.0 / np.median(inf):.2f}% of '
                     f'inference)')
        self.get_logger().info(
            f'{self._processed} classified, {self._rejected} rejected | '
            f'inference median {np.median(inf):.0f} ms -> {1000.0 / np.mean(inf):.2f} Hz '
            f'achieved | audio age at publish median {np.median(lat):.0f} ms{extra}')
        self._inference_ms.clear()
        self._latencies.clear()
        self._embed_publish_us.clear()


def main(args=None):
    """Run the contact_ml node until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = ContactMlNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
