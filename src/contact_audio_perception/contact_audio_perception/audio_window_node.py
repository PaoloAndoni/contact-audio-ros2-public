#!/usr/bin/env python3
"""
audio_window_node - assemble raw audio chunks into fixed-length analysis windows.

Subscribes /audio/raw_chunk (AudioChunk) and republishes /audio/window
(AudioWindow) as overlapping, fixed-length windows suitable for AST encoding,
optionally down-mixed to mono.

Timestamp convention (shared across the pipeline, do not deviate):
  * header.stamp        -> time of the FIRST frame in the window
  * start_sample_index  -> per-channel frame index of the window's first frame

Window timing is derived arithmetically from the capture timeline (each chunk's
start_sample_index + header.stamp), never from wall-clock arrival time, so the
window stamps stay jitter-free and line up with robot state downstream. A gap in
start_sample_index (dropped samples, or wav_player looping) or a change of audio
format triggers a re-anchor so a window is never built across a discontinuity.
"""
from contact_audio_interfaces.msg import AudioChunk, AudioWindow
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time


class AudioWindowNode(Node):
    """Turn a stream of AudioChunk messages into overlapping AudioWindow messages."""

    def __init__(self):
        """Read parameters and set up the subscription and publisher."""
        super().__init__('audio_window_node')

        self.declare_parameter('input_topic', '/audio/raw_chunk')
        self.declare_parameter('output_topic', '/audio/window')
        self.declare_parameter('window_sec', 1.0)
        self.declare_parameter('hop_sec', 0.5)
        self.declare_parameter('mono', True)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.window_sec = float(self.get_parameter('window_sec').value)
        self.hop_sec = float(self.get_parameter('hop_sec').value)
        self.mono = bool(self.get_parameter('mono').value)

        if self.window_sec <= 0.0:
            raise RuntimeError("Parameter 'window_sec' must be positive")
        if not 0.0 < self.hop_sec <= self.window_sec:
            raise RuntimeError("Parameter 'hop_sec' must be in (0, window_sec]")

        # Format + sizes are established from the first valid chunk.
        self.sample_rate = None
        self.in_channels = None
        self.out_channels = None
        self.window_frames = None
        self.hop_frames = None
        self.frame_id = 'contact_mic'
        self.buffer = np.empty(0, dtype=np.int16)  # interleaved if out_channels > 1
        self.buffer_start_index = 0                # abs frame index of buffer[0]
        self.buffer_start_stamp = None             # rclpy.time.Time of buffer[0]
        self.next_expected_index = None            # abs frame index of next chunk
        self.logged_first_window = False

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.publisher = self.create_publisher(AudioWindow, output_topic, qos)
        self.subscription = self.create_subscription(
            AudioChunk, input_topic, self.on_chunk, qos
        )
        self.get_logger().info(
            f"Windowing '{input_topic}' -> '{output_topic}' | "
            f'window={self.window_sec:.3f}s hop={self.hop_sec:.3f}s mono={self.mono}'
        )

    def _validate_chunk(self, msg):
        """Reject malformed chunks (bad rate/channels or ragged data)."""
        if msg.sample_rate <= 0:
            self.get_logger().warn(
                f'Dropping chunk: invalid sample_rate={msg.sample_rate}.',
                throttle_duration_sec=2.0,
            )
            return False
        if msg.channels <= 0:
            self.get_logger().warn(
                f'Dropping chunk: invalid channels={msg.channels}.',
                throttle_duration_sec=2.0,
            )
            return False
        if len(msg.data) % msg.channels != 0:
            self.get_logger().warn(
                f'Dropping chunk: data length {len(msg.data)} not a multiple '
                f'of channels {msg.channels}.',
                throttle_duration_sec=2.0,
            )
            return False
        return True

    def _configure_from_chunk(self, msg):
        """Set frame sizes from a chunk's format and anchor; return success."""
        sample_rate = int(msg.sample_rate)
        window_frames = int(round(self.window_sec * sample_rate))
        hop_frames = int(round(self.hop_sec * sample_rate))
        if window_frames < 1 or hop_frames < 1:
            self.get_logger().error(
                f'window_sec/hop_sec too small for {sample_rate} Hz '
                f'(window={window_frames}, hop={hop_frames} frames); shutting down.'
            )
            rclpy.shutdown()
            return False
        self.sample_rate = sample_rate
        self.in_channels = int(msg.channels)
        self.out_channels = 1 if self.mono else self.in_channels
        self.window_frames = window_frames
        self.hop_frames = hop_frames
        self._anchor(msg)
        self.get_logger().info(
            f'Anchored: {self.sample_rate} Hz, in_ch={self.in_channels}, '
            f'out_ch={self.out_channels}, window={self.window_frames} frames, '
            f'hop={self.hop_frames} frames, '
            f'overlap={self.window_frames - self.hop_frames} frames'
        )
        return True

    def _anchor(self, msg):
        """(Re)set the buffer anchor to the start of the given chunk."""
        self.buffer = np.empty(0, dtype=np.int16)
        self.buffer_start_index = int(msg.start_sample_index)
        self.buffer_start_stamp = Time.from_msg(msg.header.stamp)
        self.next_expected_index = int(msg.start_sample_index)
        self.frame_id = msg.header.frame_id or self.frame_id

    def _to_out_frames(self, msg):
        """Return the chunk samples as a 1-D int16 array in the output layout."""
        samples = np.asarray(msg.data, dtype=np.int16)
        if self.in_channels > 1:
            frames = samples.reshape(-1, self.in_channels)
            if self.mono:
                return frames.mean(axis=1).astype(np.int16).reshape(-1)
            return frames.reshape(-1)
        return samples

    def on_chunk(self, msg):
        """Accumulate a chunk and emit any windows that are now complete."""
        if not self._validate_chunk(msg):
            return

        if self.sample_rate is None:
            if not self._configure_from_chunk(msg):
                return
        elif msg.sample_rate != self.sample_rate or msg.channels != self.in_channels:
            self.get_logger().warn(
                f'Format changed ({self.sample_rate} Hz/{self.in_channels} ch -> '
                f'{msg.sample_rate} Hz/{msg.channels} ch); reconfiguring.'
            )
            if not self._configure_from_chunk(msg):
                return
        elif int(msg.start_sample_index) != self.next_expected_index:
            # Never build a window across a discontinuity -> re-anchor.
            self.get_logger().warn(
                f'Discontinuity: expected sample {self.next_expected_index}, '
                f'got {msg.start_sample_index}; re-anchoring.'
            )
            self._anchor(msg)

        block = self._to_out_frames(msg)
        self.buffer = np.concatenate((self.buffer, block))
        self.next_expected_index += block.size // self.out_channels

        window_samples = self.window_frames * self.out_channels
        hop_samples = self.hop_frames * self.out_channels
        while self.buffer.size >= window_samples:
            self._emit_window(self.buffer[:window_samples])
            self.buffer = self.buffer[hop_samples:]
            self.buffer_start_index += self.hop_frames
            self.buffer_start_stamp = self.buffer_start_stamp + Duration(
                seconds=self.hop_frames / self.sample_rate
            )

    def _emit_window(self, data):
        """Publish one AudioWindow from the given interleaved int16 samples."""
        out = AudioWindow()
        out.header.stamp = self.buffer_start_stamp.to_msg()
        out.header.frame_id = self.frame_id
        out.sample_rate = int(self.sample_rate)
        out.channels = int(self.out_channels)
        out.start_sample_index = int(self.buffer_start_index)
        out.duration_sec = float(self.window_frames / self.sample_rate)
        out.data = data.tolist()
        self.publisher.publish(out)

        if not self.logged_first_window:
            self.logged_first_window = True
            self.get_logger().info(
                f'First window: start_index={out.start_sample_index}, '
                f'frames={data.size // self.out_channels}, channels={out.channels}, '
                f'duration={out.duration_sec:.3f}s'
            )


def main(args=None):
    """Run the audio_window node until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = AudioWindowNode()
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
