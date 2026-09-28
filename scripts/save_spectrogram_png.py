"""Save one frame of /audio/spectrogram_image to a PNG.

Proof that the Foxglove image topic renders correctly, without needing Foxglove
open. Usage: python3 scripts/save_spectrogram_png.py [out.png] [timeout_sec]
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class Grabber(Node):
    """Capture a single Image message."""

    def __init__(self):
        """Subscribe to the spectrogram image topic."""
        super().__init__('save_spectrogram_png')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.frame = None
        self.create_subscription(Image, '/audio/spectrogram_image', self.on_image, qos)

    def on_image(self, msg):
        """Keep the most recent frame."""
        if msg.encoding != 'rgb8':
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        self.frame = arr.reshape(msg.height, msg.width, 3)


def main():
    """Grab one frame and write it to disk."""
    out = sys.argv[1] if len(sys.argv) > 1 else 'spectrogram_live.png'
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    rclpy.init()
    node = Grabber()
    deadline = time.time() + timeout
    while node.frame is None and time.time() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    frame = node.frame
    node.destroy_node()
    rclpy.shutdown()

    if frame is None:
        print('no image received', file=sys.stderr)
        raise SystemExit(1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.imshow(frame, aspect='auto', interpolation='nearest')
    n_mels = frame.shape[0]
    # The node flips the filterbank so low frequencies render at the bottom, so
    # image row r holds mel bin (n_mels - 1 - r). Label the mel bin, not the row.
    rows = np.linspace(0, n_mels - 1, 5).astype(int)
    ax.set_yticks(rows)
    ax.set_yticklabels([str(n_mels - 1 - r) for r in rows])
    ax.set_xlabel('time frame (10 ms hop)')
    ax.set_ylabel('mel bin')
    ax.set_title(f'/audio/spectrogram_image  -  {frame.shape[1]} x {frame.shape[0]}')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'saved {out} ({frame.shape[1]}x{frame.shape[0]})')


if __name__ == '__main__':
    main()
