"""
Publish synthetic camera frames whose CONTENT identifies the frame.

Exists so the observation assembler can be tested with no camera, and -- more
importantly -- so a test can check WHICH frames were selected rather than only
that the right NUMBER of frames arrived. Every frame carries its own sequence
number and publish timestamp encoded in its first pixels, so a verifier decodes
the image and compares against the frame it expected.

That distinction is the whole point. An assembler that always returns the two
OLDEST buffered frames, or the same frame twice, produces an array of exactly
the right shape full of perfectly plausible images. Only decoding the content
catches it.

ENCODING, little-endian, read from the raw rgb8 byte stream:
    bytes  0.. 7   frame sequence number,  uint64
    bytes  8..15   header stamp in ns,     uint64
    byte  16       MAGIC (0xA5), so a decoder can tell an encoded frame from an
                   ordinary image and refuse to "decode" the latter
The rest of the frame is a moving gradient, so a human watching in Foxglove can
see it is live.

Usage: python3 scripts/fake_camera.py [--rate 30] [--width 160] [--height 120]
"""
import argparse
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

MAGIC = 0xA5
HEADER_BYTES = 17


def decode_frame(array):
    """
    Return (sequence, stamp_ns) from an encoded frame, or raise ValueError.

    Takes the (H, W, 3) uint8 array, not the message, so it works equally on a
    live message and on an image recovered from a saved dataset.
    """
    flat = np.asarray(array, dtype=np.uint8).reshape(-1)
    if flat.size < HEADER_BYTES:
        raise ValueError(f'frame has only {flat.size} bytes, need {HEADER_BYTES}')
    raw = flat[:HEADER_BYTES].tobytes()
    if raw[16] != MAGIC:
        raise ValueError(
            f'no fake_camera magic byte (got 0x{raw[16]:02X}, expected 0x{MAGIC:02X}); '
            f'this is not an encoded frame')
    sequence, stamp_ns = struct.unpack('<QQ', raw[:16])
    return int(sequence), int(stamp_ns)


def encode_frame(height, width, sequence, stamp_ns):
    """Build one (H, W, 3) uint8 frame with its identity in the first pixels."""
    # A gradient that moves with the sequence number: visibly live in Foxglove,
    # and it guarantees two different frames are never byte-identical even if
    # the encoded header were ignored.
    row = (np.arange(width, dtype=np.int32) * 2 + sequence) % 256
    column = (np.arange(height, dtype=np.int32) * 3 + sequence) % 256
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = row[None, :]
    frame[:, :, 1] = column[:, None]
    frame[:, :, 2] = (row[None, :] + column[:, None]) % 256

    header = struct.pack('<QQ', int(sequence) & (2 ** 64 - 1),
                         int(stamp_ns) & (2 ** 64 - 1)) + bytes([MAGIC])
    flat = frame.reshape(-1)
    flat[:HEADER_BYTES] = np.frombuffer(header, dtype=np.uint8)
    return frame


class FakeCamera(Node):
    """Publish identifiable synthetic frames at a fixed rate."""

    def __init__(self, rate_hz, width, height, topic, frame_id):
        """Set up the publisher and the timer."""
        super().__init__('fake_camera')
        # Best effort, like a real driver: a camera does not retransmit.
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.publisher = self.create_publisher(Image, topic, qos)
        self.width = int(width)
        self.height = int(height)
        self.frame_id = frame_id
        self.sequence = 0
        self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(
            f'Publishing {width}x{height} rgb8 on {topic} at {rate_hz:g} Hz; each '
            f'frame carries its sequence number and stamp in its first pixels.')

    def tick(self):
        """Publish one frame."""
        now = self.get_clock().now()
        stamp_ns = now.nanoseconds
        frame = encode_frame(self.height, self.width, self.sequence, stamp_ns)

        msg = Image()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = self.width * 3
        msg.data = frame.reshape(-1).tobytes()
        self.publisher.publish(msg)

        self.sequence += 1
        if self.sequence % 300 == 0:
            self.get_logger().info(f'{self.sequence} frames published')


def main():
    """Run the fake camera until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rate', type=float, default=30.0)
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=120)
    parser.add_argument('--topic', default='/camera/image_raw')
    parser.add_argument('--frame-id', default='camera_link')
    args = parser.parse_args()

    rclpy.init()
    node = FakeCamera(args.rate, args.width, args.height, args.topic, args.frame_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
