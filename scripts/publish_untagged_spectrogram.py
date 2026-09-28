"""Publish a Spectrogram with an empty preprocessing_id.

Exists to prove the model guard rejects untagged features rather than trusting
them. Usage: python3 scripts/publish_untagged_spectrogram.py [count]
"""
import sys
import time

from contact_audio_interfaces.msg import Spectrogram
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


def main():
    """Publish a handful of well-formed but untagged spectrograms."""
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rclpy.init()
    node = Node('publish_untagged_spectrogram')
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    pub = node.create_publisher(Spectrogram, '/audio/spectrogram', qos)
    for _ in range(20):
        if pub.get_subscription_count() > 0:
            break
        rclpy.spin_once(node, timeout_sec=0.1)

    for i in range(count):
        msg = Spectrogram()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'contact_mic'
        msg.start_sample_index = i * 24000
        msg.source_sample_rate = 48000
        msg.analysis_sample_rate = 16000
        msg.window_duration_sec = 1.0
        msg.frame_length_ms = 25.0
        msg.frame_shift_ms = 10.0
        msg.peak_gain = 1.0
        msg.n_frames = 98
        msg.n_mels = 128
        msg.feature_type = 'kaldi_fbank_v1'
        msg.preprocessing_id = ''          # the whole point of this script
        msg.data = np.zeros(98 * 128, dtype=np.float32)
        pub.publish(msg)
        time.sleep(0.5)
    print(f'published {count} untagged spectrograms')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
