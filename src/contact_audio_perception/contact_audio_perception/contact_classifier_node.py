#!/usr/bin/env python3
"""
contact_classifier_node - classify AST embeddings into contact types.

Subscribes /audio/ast_embedding (AstEmbedding), runs a nearest-centroid
classifier loaded from a centroids .npz (see train_centroids.py), and publishes
/audio/contact_prediction (ContactPrediction). The embedding's header (window
start stamp) is propagated so predictions stay time-aligned with the audio and,
later, robot state.
"""
from contact_audio_interfaces.msg import AstEmbedding, ContactPrediction
from contact_audio_perception.contact_classifier import CentroidClassifier
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class ContactClassifierNode(Node):
    """Classify incoming AstEmbedding messages into ContactPrediction messages."""

    def __init__(self):
        """Load the centroid model and set up the subscription and publisher."""
        super().__init__('contact_classifier_node')

        self.declare_parameter('input_topic', '/audio/ast_embedding')
        self.declare_parameter('output_topic', '/audio/contact_prediction')
        self.declare_parameter('model_path', '')
        self.declare_parameter('unknown_threshold', 0.0)
        self.declare_parameter('temperature', 0.1)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        model_path = self.get_parameter('model_path').value
        unknown_threshold = float(self.get_parameter('unknown_threshold').value)
        temperature = float(self.get_parameter('temperature').value)

        if not model_path:
            self.get_logger().error(
                "Parameter 'model_path' is required (a centroids .npz produced "
                'by train_centroids.py).'
            )
            raise SystemExit(1)
        try:
            self.classifier = CentroidClassifier.from_npz(
                model_path, unknown_threshold, temperature
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to load model {model_path!r}: {exc}')
            raise SystemExit(1) from exc

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.publisher = self.create_publisher(ContactPrediction, output_topic, qos)
        self.subscription = self.create_subscription(
            AstEmbedding, input_topic, self.on_embedding, qos
        )
        self.get_logger().info(
            f"Classifying '{input_topic}' -> '{output_topic}' | "
            f'classes={self.classifier.class_names}'
        )

    def on_embedding(self, msg):
        """Classify one embedding and publish the contact prediction."""
        expected = self.classifier.centroids.shape[1]
        if len(msg.cls) != expected:
            self.get_logger().warn(
                f'Dropping embedding: dim {len(msg.cls)} != model dim {expected}.',
                throttle_duration_sec=2.0,
            )
            return

        label, confidence, class_names, probs = self.classifier.predict(msg.cls)

        out = ContactPrediction()
        out.header = msg.header
        out.start_sample_index = int(msg.start_sample_index)
        out.label = label
        out.confidence = confidence
        out.class_names = class_names
        out.probabilities = probs
        self.publisher.publish(out)
        self.get_logger().info(
            f'{label} ({confidence:.2f}) @ start_index={msg.start_sample_index}',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    """Run the contact_classifier node until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = ContactClassifierNode()
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
