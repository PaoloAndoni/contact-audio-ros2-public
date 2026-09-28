#!/usr/bin/env python3
"""
audio_capture_node - stream a live microphone onto /audio/raw_chunk.

Real-acquisition sibling of wav_player_node: publishes the SAME AudioChunk
message (int16, timestamped, sample-indexed) so the rest of the pipeline
(audio_window -> ast_encoder -> contact_classifier) runs unchanged.

Uses python3-sounddevice (bundled PortAudio). The audio callback only copies
samples into a queue; a ROS timer drains it and publishes, so the realtime audio
thread never blocks on ROS work.

Timestamp convention (shared across the pipeline):
  * header.stamp        -> time of the FIRST frame in the chunk (anchored at the
                           first callback, advanced by cumulative frame count)
  * start_sample_index  -> per-channel frame index (0 at stream start)
  * data                -> interleaved int16 if channels > 1
"""
import queue

from contact_audio_interfaces.msg import AudioChunk
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


def _resolve_device(value):
    """Map the 'device' param to a sounddevice selector (None / int / name)."""
    if value is None or value == '':
        return None
    text = str(value)
    return int(text) if text.lstrip('-').isdigit() else text


class AudioCaptureNode(Node):
    """Capture live microphone audio and publish it as AudioChunk messages."""

    def __init__(self):
        """Open the input stream and start publishing captured chunks."""
        super().__init__('audio_capture_node')

        self.declare_parameter('device', '')
        self.declare_parameter('sample_rate', 48000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('chunk_size', 2048)
        self.declare_parameter('topic', '/audio/raw_chunk')
        self.declare_parameter('frame_id', 'contact_mic')
        self.declare_parameter('list_devices', False)

        device = _resolve_device(self.get_parameter('device').value)
        self.sample_rate = int(self.get_parameter('sample_rate').value)
        self.channels = int(self.get_parameter('channels').value)
        self.chunk_size = int(self.get_parameter('chunk_size').value)
        topic = self.get_parameter('topic').value
        self.frame_id = self.get_parameter('frame_id').value

        try:
            import sounddevice as sd
        except Exception as exc:
            self.get_logger().error(
                f'python3-sounddevice is required for audio_capture_node: {exc}. '
                f'Install: pip install --user --break-system-packages sounddevice'
            )
            raise SystemExit(1) from exc

        if bool(self.get_parameter('list_devices').value):
            self.get_logger().info('Input devices:\n' + str(sd.query_devices()))
            raise SystemExit(0)

        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.publisher = self.create_publisher(AudioChunk, topic, qos)

        self._queue = queue.Queue()
        self._captured = 0      # frames handed to the callback so far
        self.start_time = None  # anchored on the first callback

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=self.channels,
                dtype='int16',
                device=device,
                callback=self._on_audio,
            )
            self.stream.start()
        except Exception as exc:
            self.get_logger().error(
                f'Failed to open input device {device!r} '
                f'({self.sample_rate} Hz, {self.channels} ch): {exc}. '
                f'Run with -p list_devices:=true to see available devices.'
            )
            raise SystemExit(1) from exc

        period = self.chunk_size / self.sample_rate / 2.0
        self.timer = self.create_timer(period, self._drain)
        self.get_logger().info(
            f"Capturing device {device!r} -> '{topic}' | "
            f'{self.sample_rate} Hz, {self.channels} ch, chunk={self.chunk_size}'
        )

    def _on_audio(self, indata, frames, time_info, status):
        """Copy captured samples and their frame index into the queue (audio thread)."""
        if status:
            self.get_logger().warn(str(status), throttle_duration_sec=2.0)
        if self.start_time is None:
            self.start_time = self.get_clock().now()
        self._queue.put((indata.copy(), self._captured))
        self._captured += frames

    def _drain(self):
        """Publish any captured chunks waiting in the queue."""
        while True:
            try:
                block, start_index = self._queue.get_nowait()
            except queue.Empty:
                return
            msg = AudioChunk()
            stamp = self.start_time + Duration(seconds=start_index / self.sample_rate)
            msg.header.stamp = stamp.to_msg()
            msg.header.frame_id = self.frame_id
            msg.sample_rate = self.sample_rate
            msg.channels = self.channels
            msg.start_sample_index = int(start_index)
            msg.data = block.reshape(-1).tolist()
            self.publisher.publish(msg)

    def destroy_node(self):
        """Stop and close the audio stream."""
        stream = getattr(self, 'stream', None)
        if stream is not None:
            stream.stop()
            stream.close()
        super().destroy_node()


def main(args=None):
    """Run the audio_capture node until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = AudioCaptureNode()
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
