#!/usr/bin/env python3
"""
ast_encoder_node - encode audio windows into AST CLS embeddings.

Subscribes /audio/window (AudioWindow), preprocesses to mono 16 kHz, runs the
AST model, and publishes /audio/ast_embedding (AstEmbedding) carrying the 768-D
CLS token. The window's header.stamp + start_sample_index are propagated
unchanged, so the embedding stays aligned to when the sound happened.

The model is loaded once at startup (first run downloads ~350 MB and blocks). The
window subscription uses best-effort, depth-1 QoS so that if inference lags, stale
windows are dropped rather than queued (latest window wins) - keeping the live
path real-time.
"""
from contact_audio_interfaces.msg import AstEmbedding, AudioWindow
from contact_audio_perception.ast_encoder import (
    AstEncoder,
    DEFAULT_MODEL_ID,
    window_to_model_input,
)
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class AstEncoderNode(Node):
    """Encode incoming AudioWindow messages into AstEmbedding messages."""

    def __init__(self):
        """Load the AST model and set up the subscription and publisher."""
        super().__init__('ast_encoder_node')

        self.declare_parameter('input_topic', '/audio/window')
        self.declare_parameter('output_topic', '/audio/ast_embedding')
        self.declare_parameter('model_id', DEFAULT_MODEL_ID)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        model_id = self.get_parameter('model_id').value

        self.get_logger().info(
            f'Loading AST model {model_id} '
            f'(first run downloads ~350 MB from Hugging Face and blocks)...'
        )
        try:
            self.encoder = AstEncoder(model_id)
        except Exception as exc:
            self.get_logger().error(
                f'Failed to load AST model {model_id!r}: {exc}. Ensure it is '
                f'cached under ~/.cache/huggingface or the machine has network '
                f'access (first load downloads ~350 MB).'
            )
            raise SystemExit(1) from exc
        self.get_logger().info('AST model ready.')

        # Best-effort depth-1: drop stale windows if inference falls behind.
        sub_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        pub_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.publisher = self.create_publisher(AstEmbedding, output_topic, pub_qos)
        self.subscription = self.create_subscription(
            AudioWindow, input_topic, self.on_window, sub_qos
        )
        self.logged_first = False
        self.get_logger().info(f"Encoding '{input_topic}' -> '{output_topic}'")

    def _valid_window(self, msg):
        """Reject malformed windows (bad rate/channels or ragged/empty data)."""
        if msg.sample_rate <= 0:
            self.get_logger().warn(
                f'Dropping window: invalid sample_rate={msg.sample_rate}.',
                throttle_duration_sec=2.0,
            )
            return False
        if msg.channels <= 0:
            self.get_logger().warn(
                f'Dropping window: invalid channels={msg.channels}.',
                throttle_duration_sec=2.0,
            )
            return False
        if len(msg.data) == 0 or len(msg.data) % msg.channels != 0:
            self.get_logger().warn(
                f'Dropping window: data length {len(msg.data)} is not a positive '
                f'multiple of channels {msg.channels}.',
                throttle_duration_sec=2.0,
            )
            return False
        return True

    def on_window(self, msg):
        """Encode a window and publish its CLS embedding."""
        if not self._valid_window(msg):
            return

        audio = window_to_model_input(msg.data, int(msg.channels), int(msg.sample_rate))
        cls = self.encoder.encode(audio)

        out = AstEmbedding()
        out.header = msg.header
        out.model_id = self.encoder.model_id
        out.start_sample_index = int(msg.start_sample_index)
        out.window_duration_sec = float(msg.duration_sec)
        out.cls = cls.tolist()
        self.publisher.publish(out)

        if not self.logged_first:
            self.logged_first = True
            self.get_logger().info(
                f'First embedding: dim={len(out.cls)}, '
                f'start_index={out.start_sample_index}'
            )


def main(args=None):
    """Run the ast_encoder node until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = AstEncoderNode()
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
