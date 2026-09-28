#!/usr/bin/env python3
"""
wav_player_node - replays a WAV file onto /audio/raw_chunk as AudioChunk.

Phase-1 stand-in for audio_capture_node so the whole graph can be built and
validated from files (no live mic needed). Drop-in sibling of the future
live-mic node: same topic, same AudioChunk message.

Timestamp convention (shared across the pipeline, do not deviate):
  * header.stamp        -> time of the FIRST frame in this chunk
  * start_sample_index  -> per-channel frame index (0 at file start);
                           seconds = start_sample_index / sample_rate
  * data                -> interleaved int16 (L R L R ...) when channels > 1
"""
import array
import os
import sys
import wave

from contact_audio_interfaces.msg import AudioChunk
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class WavPlayerNode(Node):
    """Stream a WAV file in fixed-size chunks as timestamped AudioChunk messages."""

    def __init__(self):
        """Load and validate the WAV file, then start the publish timer."""
        super().__init__('wav_player_node')

        self.declare_parameter('wav_path', '')
        self.declare_parameter('topic', '/audio/raw_chunk')
        self.declare_parameter('chunk_size', 2048)   # frames per message
        self.declare_parameter('loop', False)
        self.declare_parameter('realtime', True)
        self.declare_parameter('frame_id', 'contact_mic')
        self.declare_parameter('wait_for_subscriber', True)
        self.declare_parameter('shutdown_when_done', True)

        self.wav_path = self.get_parameter('wav_path').value
        topic = self.get_parameter('topic').value
        self.chunk_size = int(self.get_parameter('chunk_size').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.realtime = bool(self.get_parameter('realtime').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.wait_for_subscriber = bool(self.get_parameter('wait_for_subscriber').value)
        self.shutdown_when_done = bool(self.get_parameter('shutdown_when_done').value)

        if not self.wav_path:
            raise RuntimeError("Parameter 'wav_path' is required")
        if not os.path.isfile(self.wav_path):
            raise RuntimeError(f'WAV file does not exist: {self.wav_path}')
        if self.chunk_size <= 0:
            raise RuntimeError("Parameter 'chunk_size' must be greater than 0")

        # Open first, then validate; close the handle before raising so a bad
        # file never leaks an open descriptor.
        self.wav_file = None
        wav_file = wave.open(self.wav_path, 'rb')
        sampwidth = wav_file.getsampwidth()
        if sampwidth != 2:
            wav_file.close()
            raise RuntimeError(
                'Only 16-bit PCM WAV files are supported '
                f'(got {sampwidth * 8}-bit samples)'
            )
        self.sample_rate = wav_file.getframerate()
        self.channels = wav_file.getnchannels()
        self.n_frames = wav_file.getnframes()
        self.wav_file = wav_file

        # Reliable + buffered: dataset correctness needs every sample delivered.
        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.publisher = self.create_publisher(AudioChunk, topic, qos)

        # Wait for a subscriber so the first (volatile-QoS) chunks are not dropped
        # during discovery -> a downstream window node can anchor at sample 0.
        if self.wait_for_subscriber:
            self._await_subscriber()

        self.cursor = 0  # per-channel frame index of the next frame to publish
        self.start_time = self.get_clock().now()
        period = (self.chunk_size / self.sample_rate) if self.realtime else 1.0e-3
        self.timer = self.create_timer(period, self.publish_next_chunk)

        self.get_logger().info(
            f"Publishing {self.wav_path} on '{topic}' | {self.sample_rate} Hz, "
            f'{self.channels} ch, {self.n_frames} frames '
            f'({self.n_frames / self.sample_rate:.2f} s), chunk={self.chunk_size}, '
            f'realtime={self.realtime}, loop={self.loop}'
        )

    def _await_subscriber(self, max_ticks=50):
        """Spin briefly until a subscriber connects (bounded, ~5 s)."""
        for _ in range(max_ticks):
            if self.publisher.get_subscription_count() > 0:
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn('No subscriber connected; publishing anyway.')

    def publish_next_chunk(self):
        """Read the next chunk from the file and publish it as an AudioChunk."""
        frames = self.wav_file.readframes(self.chunk_size)
        if not frames:
            if self.loop:
                self.wav_file.rewind()
                self.cursor = 0
                self.start_time = self.get_clock().now()
                return
            self.get_logger().info('Finished publishing WAV file')
            self.timer.cancel()
            if self.shutdown_when_done:
                rclpy.shutdown()
            return

        samples = array.array('h')
        samples.frombytes(frames)
        if sys.byteorder != 'little':
            samples.byteswap()  # WAV is little-endian; publish host-order int16

        msg = AudioChunk()
        stamp = self.start_time + Duration(seconds=self.cursor / self.sample_rate)
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.frame_id
        msg.sample_rate = int(self.sample_rate)
        msg.channels = int(self.channels)
        msg.start_sample_index = int(self.cursor)
        msg.data = samples.tolist()

        self.publisher.publish(msg)
        self.cursor += len(samples) // self.channels  # advance by frames, not samples

    def destroy_node(self):
        """Close the WAV file, if open, and destroy the node."""
        if getattr(self, 'wav_file', None) is not None:
            self.wav_file.close()
        super().destroy_node()


def main(args=None):
    """Run the wav_player node until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = WavPlayerNode()
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
