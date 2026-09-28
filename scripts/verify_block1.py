"""Subscribe to the DAQ node's output and report whether Block 1 works.

Usage: python3 scripts/verify_block1.py [seconds]
"""
import sys
import time

from contact_audio_interfaces.msg import Spectrogram
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class Checker(Node):
    """Collect statistics on /audio/spectrogram and /audio/spectrogram_image."""

    def __init__(self):
        """Subscribe to both DAQ outputs."""
        super().__init__('verify_block1')
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.first = None
        self.records = []
        self.arrivals = []
        self.images = []
        self.create_subscription(Spectrogram, '/audio/spectrogram', self.on_spec, qos)
        self.create_subscription(Image, '/audio/spectrogram_image', self.on_image, qos)

    def on_spec(self, msg):
        """Record one spectrogram's metadata and when it arrived.

        Deliberately does NOT retain msg.data: holding every 12544-float payload
        makes this checker stall, which back-pressures the publisher's KEEP_LAST
        history into dropping messages -- and the dropped messages then look
        exactly like a publisher stall. Keep the instrument cheap.
        """
        if self.first is None:
            self.first = msg
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.records.append((stamp, msg.start_sample_index))
        self.arrivals.append(time.monotonic())

    def on_image(self, msg):
        """Record one image message."""
        self.images.append((msg.height, msg.width, msg.encoding, len(msg.data)))


def main():
    """Spin for a while, then print a pass/fail report."""
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    rclpy.init()
    node = Checker()
    deadline = time.time() + duration
    while time.time() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    records, images = node.records, node.images
    print(f'spectrogram messages : {len(records)}')
    print(f'image messages       : {len(images)}')
    if not records:
        print('RESULT: FAIL - nothing received')
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

    first = node.first
    stamps = [r[0] for r in records]
    idx = [r[1] for r in records]
    deltas = [b - a for a, b in zip(stamps, stamps[1:])]
    steps = [b - a for a, b in zip(idx, idx[1:])]

    print(f'shape                : {first.n_frames} x {first.n_mels} (time-major)')
    print(f'data length          : {len(first.data)} '
          f'(expected {first.n_frames * first.n_mels})')
    print(f'source / analysis    : {first.source_sample_rate} Hz -> '
          f'{first.analysis_sample_rate} Hz')
    print(f'framing              : {first.frame_length_ms:g} / {first.frame_shift_ms:g} ms')
    print(f'feature_type         : {first.feature_type}')
    print(f'preprocessing_id     : {first.preprocessing_id}')
    print(f'peak_gain (first)    : {first.peak_gain:.6f}')
    if deltas:
        mean_dt = sum(deltas) / len(deltas)
        print(f'stamp step           : mean {mean_dt * 1000:.2f} ms '
              f'-> {1.0 / mean_dt:.2f} Hz (min {min(deltas) * 1000:.2f}, '
              f'max {max(deltas) * 1000:.2f})')
    if steps:
        print(f'sample-index step    : min {min(steps)} max {max(steps)} '
              f'(expected {first.source_sample_rate / 15.0:.0f})')
    # A gap that appears in BOTH arrival and stamp time is the publisher stalling;
    # a gap in stamp time only means audio arrived late and then burst in.
    gaps = [b - a for a, b in zip(node.arrivals, node.arrivals[1:])]
    if gaps:
        print(f'arrival step         : mean {sum(gaps) / len(gaps) * 1000:.2f} ms '
              f'(min {min(gaps) * 1000:.2f}, max {max(gaps) * 1000:.2f})')
        worst = max(range(len(deltas)), key=lambda i: deltas[i]) if deltas else None
        if worst is not None:
            print(f'worst stamp gap      : {deltas[worst] * 1000:.1f} ms of audio, '
                  f'delivered {gaps[worst] * 1000:.1f} ms apart in wall clock')
    if images:
        h, w, enc, n = images[0]
        print(f'image                : {w}x{h} {enc}, {n} bytes')

    ok = True
    ok &= len(first.data) == first.n_frames * first.n_mels
    ok &= first.n_mels == 128
    ok &= first.analysis_sample_rate == 16000
    ok &= bool(first.preprocessing_id)
    ok &= all(s > 0 for s in steps) if steps else True
    if images:
        h, w, enc, n = images[0]
        ok &= (h == first.n_mels and w == first.n_frames and enc == 'rgb8' and n == h * w * 3)
    print('RESULT:', 'PASS' if ok else 'FAIL')

    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
