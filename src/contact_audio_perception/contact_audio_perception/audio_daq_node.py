#!/usr/bin/env python3
"""
audio_daq_node - acquisition + framing + filterbank in one node.

Replaces audio_capture_node, audio_window_node and wav_player_node: it owns a
rolling buffer of the last `buffer_sec` seconds of waveform and republishes it
as mel filterbank features `publish_rate_hz` times a second, plus an image of
the same features for Foxglove.

    source=mic|wav -> ring buffer -> /audio/spectrogram        (Spectrogram)
                                  -> /audio/spectrogram_image  (sensor_msgs/Image)

Timestamp convention (shared across the pipeline, do not deviate):
  * header.stamp       -> time of the FIRST sample in the published buffer
  * start_sample_index -> per-channel frame index of that first sample

Buffers overlap heavily by design: at 15 Hz with a 1 s buffer, consecutive
messages share 93% of their audio. The features are published UNPADDED and
UNNORMALISED -- see spectrogram.py and Spectrogram.msg for why.
"""
import queue
import threading
import wave

from contact_audio_interfaces.msg import Spectrogram
from contact_audio_perception.spectrogram import (
    ANALYSIS_SAMPLE_RATE,
    buffer_to_fbank,
    DEFAULT_FRAME_LENGTH_MS,
    DEFAULT_FRAME_SHIFT_MS,
    DEFAULT_N_MELS,
    FbankConfig,
    FEATURE_TYPE,
)
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image

# Viridis anchors; enough stops that linear interpolation between them is
# visually indistinguishable from the real colormap, without pulling matplotlib
# into a node that has to keep up with a 15 Hz timer.
_VIRIDIS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [253, 231, 37],
], dtype=np.float32)


def _viridis_lut():
    """Build a 256-entry uint8 RGB lookup table."""
    stops = np.linspace(0.0, 1.0, len(_VIRIDIS))
    grid = np.linspace(0.0, 1.0, 256)
    channels = [np.interp(grid, stops, _VIRIDIS[:, c]) for c in range(3)]
    return np.stack(channels, axis=1).astype(np.uint8)


def _resolve_device(value):
    """Map the 'device' parameter to a sounddevice selector (None / int / name)."""
    if value is None or value == '':
        return None
    text = str(value)
    return int(text) if text.lstrip('-').isdigit() else text


class AudioDaqNode(Node):
    """Acquire audio and publish rolling mel filterbank features."""

    def __init__(self):
        """Read parameters, open the source, and start the publish timer."""
        super().__init__('audio_daq_node')

        self.declare_parameter('source', 'mic')
        self.declare_parameter('device', '')
        self.declare_parameter('list_devices', False)
        self.declare_parameter('wav_path', '')
        self.declare_parameter('loop', False)
        self.declare_parameter('shutdown_when_done', True)

        self.declare_parameter('source_sample_rate', 0)   # 0 = ask the device
        self.declare_parameter('channels', 0)             # 0 = ask the device
        # Blocksize sets how finely the buffer can advance: the published content
        # moves in blocksize/rate steps, so 2048 @ 44.1 kHz would quantise a 15 Hz
        # stream to 46 ms jumps. 1024 halves that at negligible cost.
        self.declare_parameter('blocksize', 1024)
        self.declare_parameter('buffer_sec', 1.0)
        self.declare_parameter('publish_rate_hz', 15.0)
        self.declare_parameter('wav_feed_period_sec', 0.02)
        self.declare_parameter('wav_realtime', True)

        self.declare_parameter('analysis_sample_rate', ANALYSIS_SAMPLE_RATE)
        self.declare_parameter('n_mels', DEFAULT_N_MELS)
        self.declare_parameter('frame_length_ms', DEFAULT_FRAME_LENGTH_MS)
        self.declare_parameter('frame_shift_ms', DEFAULT_FRAME_SHIFT_MS)
        self.declare_parameter('peak_normalise', True)
        self.declare_parameter('legacy_int16_downmix', True)

        self.declare_parameter('topic', '/audio/spectrogram')
        self.declare_parameter('image_topic', '/audio/spectrogram_image')
        self.declare_parameter('publish_image', True)
        self.declare_parameter('image_min', -15.0)
        self.declare_parameter('image_max', 5.0)
        self.declare_parameter('frame_id', 'contact_mic')
        self.declare_parameter('timing_log_path', '')   # CSV of per-callback timing
        self.declare_parameter('timing_residual_warn_ms', 5.0)
        self.declare_parameter('torch_threads', 1)

        # A 98x128 filterbank is far too small to parallelise. Left at torch's
        # default this node burned 18 ms of CPU per 3 ms of work -- six cores to
        # do one core's job -- because the threads spin-wait between calls, and
        # that contention also showed up as periodically late publish ticks.
        # Single-threaded is both FASTER in wall time and ~5x cheaper. Matters
        # doubly on the robot, where the DAQ must not starve the control stack.
        torch_threads = int(self.get_parameter('torch_threads').value)
        if torch_threads > 0:
            import torch
            torch.set_num_threads(torch_threads)

        self.source = str(self.get_parameter('source').value).lower()
        self.buffer_sec = float(self.get_parameter('buffer_sec').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.wav_feed_period_sec = float(self.get_parameter('wav_feed_period_sec').value)
        self.wav_realtime = bool(self.get_parameter('wav_realtime').value)
        self.timing_residual_warn_ms = float(
            self.get_parameter('timing_residual_warn_ms').value)
        self.blocksize = int(self.get_parameter('blocksize').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_image = bool(self.get_parameter('publish_image').value)
        self.image_min = float(self.get_parameter('image_min').value)
        self.image_max = float(self.get_parameter('image_max').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.shutdown_when_done = bool(self.get_parameter('shutdown_when_done').value)

        if self.source not in ('mic', 'wav'):
            raise RuntimeError(f"Parameter 'source' must be 'mic' or 'wav', got {self.source!r}")
        if self.buffer_sec <= 0.0:
            raise RuntimeError("Parameter 'buffer_sec' must be positive")
        if self.publish_rate_hz <= 0.0:
            raise RuntimeError("Parameter 'publish_rate_hz' must be positive")
        if self.wav_feed_period_sec <= 0.0:
            raise RuntimeError("Parameter 'wav_feed_period_sec' must be positive")
        if self.image_max <= self.image_min:
            raise RuntimeError("Parameter 'image_max' must exceed 'image_min'")

        self.config = FbankConfig(
            analysis_sample_rate=self.get_parameter('analysis_sample_rate').value,
            n_mels=self.get_parameter('n_mels').value,
            frame_length_ms=self.get_parameter('frame_length_ms').value,
            frame_shift_ms=self.get_parameter('frame_shift_ms').value,
            peak_normalise=self.get_parameter('peak_normalise').value,
            legacy_int16_downmix=self.get_parameter('legacy_int16_downmix').value,
        )

        # Ring buffer state, guarded because the audio thread also writes to it.
        self._lock = threading.Lock()
        self._ring = np.empty(0, dtype=np.int16)   # interleaved, newest at the end
        self._frames_written = 0                   # per-channel frames ever fed
        self._t0 = None                            # clock time of frame 0
        self._anchor_ns = None                     # ROS ns of _anchor_index
        self._anchor_index = 0                     # frame index the anchor refers to
        self._adc_time_missing = 0
        self._status_events = 0
        self._future_stamps = 0
        self._timing_log = None
        self._timing_discontinuities = 0
        self._worst_residual_ms = 0.0
        self._last_residual_ms = 0.0
        self._queue = queue.Queue()
        self._stream = None
        self._wav = None
        self._eof = False
        self._logged_first = False
        self._last_index = None                    # guards against republishing a buffer
        self._last_tick = None
        self._late_ticks = 0

        if self.source == 'mic':
            self._open_mic()
        else:
            self._open_wav()

        self.buffer_frames = int(round(self.buffer_sec * self.source_sample_rate))
        if self.buffer_frames < 1:
            raise RuntimeError('buffer_sec is too small for the source sample rate')
        # Frames of audio between consecutive publishes.
        self._hop_frames = max(1, int(round(
            self.source_sample_rate / self.publish_rate_hz)))
        self._frames_fed = 0
        self._feed_started_at = None
        self._lut = _viridis_lut()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.publisher = self.create_publisher(Spectrogram, self.get_parameter('topic').value, qos)
        self.image_publisher = None
        if self.publish_image:
            self.image_publisher = self.create_publisher(
                Image, self.get_parameter('image_topic').value, qos)

        expected = self.config.frames_for(self.buffer_sec)
        self.get_logger().info(
            f'DAQ source={self.source} {self.source_sample_rate} Hz {self.channels} ch | '
            f'buffer={self.buffer_sec:.3f}s ({self.buffer_frames} frames) | '
            f'publishing {self.publish_rate_hz:.1f} Hz')
        self.get_logger().info(
            f'Features: {self.config.describe()} -> expected {expected} x {self.config.n_mels}')

        timing_path = self.get_parameter('timing_log_path').value
        if timing_path:
            self._timing_log = open(timing_path, 'w')
            self._timing_log.write(
                'callback_ros_ns,input_buffer_adc_time,current_time,'
                'block_start_index,derived_first_sample_ns,block_frames,status,'
                'residual_ms\n')
            self.get_logger().info(f'Writing per-callback timing to {timing_path}')

        if self.source == 'wav' and self.wav_realtime:
            self.feed_timer = self.create_timer(self.wav_feed_period_sec, self._feed_wav)
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._tick)

    # ---- sources -------------------------------------------------------

    def _open_mic(self):
        """Open the input stream, taking the device's own rate unless overridden."""
        try:
            import sounddevice as sd
        except Exception as exc:
            self.get_logger().error(
                f'python3-sounddevice is required for source=mic: {exc}. Install with '
                f'pip install --user --break-system-packages sounddevice, and ensure '
                f'libportaudio2 is present.')
            raise SystemExit(1) from exc

        device = _resolve_device(self.get_parameter('device').value)
        if bool(self.get_parameter('list_devices').value):
            self.get_logger().info('Input devices:\n' + str(sd.query_devices()))
            raise SystemExit(0)

        # Do not hardcode a rate: WSLg's RDPSource is 44100 Hz while the recorded
        # dataset is 48000 Hz, and asking for the wrong one inserts a resample we
        # do not control.
        requested_rate = int(self.get_parameter('source_sample_rate').value)
        requested_channels = int(self.get_parameter('channels').value)
        try:
            info = sd.query_devices(device, 'input')
            default_rate = int(round(float(info['default_samplerate'])))
            max_channels = int(info['max_input_channels'])
        except Exception as exc:
            self.get_logger().error(
                f'Cannot query input device {device!r}: {exc}. '
                f'Run with -p list_devices:=true to see what is available.')
            raise SystemExit(1) from exc

        self.source_sample_rate = requested_rate if requested_rate > 0 else default_rate
        self.channels = requested_channels if requested_channels > 0 else min(1, max_channels)
        if self.channels < 1:
            self.get_logger().error(f'Device {device!r} reports no input channels.')
            raise SystemExit(1)

        try:
            self._stream = sd.InputStream(
                samplerate=self.source_sample_rate, blocksize=self.blocksize,
                channels=self.channels, dtype='int16', device=device,
                callback=self._on_audio)
            self._stream.start()
        except Exception as exc:
            self.get_logger().error(
                f'Failed to open input device {device!r} at {self.source_sample_rate} Hz, '
                f'{self.channels} ch: {exc}. Run with -p list_devices:=true to list devices.')
            raise SystemExit(1) from exc

    def _open_wav(self):
        """Open and validate the WAV file used as a reproducible offline source."""
        path = self.get_parameter('wav_path').value
        if not path:
            raise RuntimeError("Parameter 'wav_path' is required when source is 'wav'")
        wav = wave.open(path, 'rb')
        if wav.getsampwidth() != 2:
            wav.close()
            raise RuntimeError(
                f'Only 16-bit PCM WAV files are supported (got {wav.getsampwidth() * 8}-bit)')
        self.source_sample_rate = wav.getframerate()
        self.channels = wav.getnchannels()
        self._wav = wav
        self.get_logger().info(
            f'Replaying {path}: {wav.getnframes()} frames '
            f'({wav.getnframes() / self.source_sample_rate:.2f} s)')

    def _on_audio(self, indata, frames, time_info, status):
        """Hand captured samples to the ROS thread (runs on the audio thread)."""
        # PortAudio reports when the FIRST SAMPLE hit the ADC. Using the callback
        # execution time instead is wrong by however long the samples waited: on
        # a startup burst many buffers arrive faster than real time, and adding
        # their sample counts to a late anchor pushes window stamps into the
        # future (a negative measured age). Map PortAudio's clock onto the ROS
        # clock per callback and keep the ADC time of the first sample.
        ros_now = self.get_clock().now().nanoseconds
        adc = float(getattr(time_info, 'inputBufferAdcTime', 0.0) or 0.0)
        current = float(getattr(time_info, 'currentTime', 0.0) or 0.0)
        if adc > 0.0 and current > 0.0:
            first_sample_ns = int(round(adc * 1e9)) + (ros_now - int(round(current * 1e9)))
        else:
            # Some backends do not populate ADC time. Fall back, but record it:
            # a silent fallback would make the timing gate meaningless.
            first_sample_ns = ros_now
            self._adc_time_missing += 1
        if status:
            self._status_events += 1
        self._queue.put((first_sample_ns, indata.copy(),
                         (ros_now, adc, current, str(status) if status else '')))

    # ---- buffering -----------------------------------------------------

    def _feed(self, block, first_sample_ns=None):
        """Append interleaved int16 samples and re-anchor the sample->time map."""
        if block.size == 0:
            return
        with self._lock:
            index = self._frames_written
            if first_sample_ns is None:
                # File replay has no hardware clock: anchor once and derive.
                if self._t0 is None:
                    self._t0 = self.get_clock().now().nanoseconds
                first_sample_ns = self._t0 + int(round(index / self.source_sample_rate
                                                       * 1e9))
            elif self._t0 is None:
                self._t0 = first_sample_ns
            # Compare the hardware timestamp against what the sample counter
            # predicts. On a sound device these agree to well under a
            # millisecond; a persistent disagreement means samples are being
            # dropped or the two clocks run at different rates, and either way
            # the stamps stop meaning "when the sound happened".
            if self._anchor_ns is not None and first_sample_ns is not None:
                predicted = self._anchor_ns + int(round(
                    (index - self._anchor_index) / self.source_sample_rate * 1e9))
                residual_ms = (first_sample_ns - predicted) / 1e6
                self._last_residual_ms = residual_ms
                if abs(residual_ms) > self.timing_residual_warn_ms:
                    self._timing_discontinuities += 1
                    self._worst_residual_ms = max(
                        self._worst_residual_ms, abs(residual_ms))
            # Anchor on the MOST RECENT measurement rather than on frame 0, so a
            # sample-clock/ROS-clock divergence cannot accumulate across a run.
            self._anchor_ns = first_sample_ns
            self._anchor_index = index
            self._ring = np.concatenate((self._ring, block))
            self._frames_written += block.size // self.channels
            excess = self._ring.size - self.buffer_frames * self.channels
            if excess > 0:
                self._ring = self._ring[excess:]

    def _pump_source(self):
        """Move whatever the microphone has produced into the ring buffer."""
        while True:
            try:
                first_sample_ns, block, diag = self._queue.get_nowait()
            except queue.Empty:
                return
            index = self._frames_written
            self._feed(np.asarray(block, dtype=np.int16).reshape(-1), first_sample_ns)
            if self._timing_log is not None:
                ros_now, adc, current, status = diag
                self._timing_log.write(
                    f'{ros_now},{adc!r},{current!r},{index},{first_sample_ns},'
                    f'{block.size // self.channels},{status},'
                    f'{self._last_residual_ms:.4f}\n')
            if self._timing_discontinuities:
                self.get_logger().warn(
                    f'Capture timing: {self._timing_discontinuities} blocks where '
                    f'the hardware timestamp disagreed with the sample counter by '
                    f'more than {self.timing_residual_warn_ms:g} ms (worst '
                    f'{self._worst_residual_ms:.0f} ms). Samples are being dropped '
                    f'or the clocks run at different rates; window stamps are only '
                    f'as good as this.', throttle_duration_sec=15.0)

    def _read_wav(self, frames):
        """Read and buffer up to `frames` frames, handling loop and EOF."""
        if self._eof or frames < 1:
            return
        raw = self._wav.readframes(frames)
        if not raw:
            if self.loop:
                self._wav.rewind()
                return
            self._eof = True
            self.get_logger().info('Finished replaying WAV file')
            return
        block = np.frombuffer(raw, dtype='<i2').astype(np.int16)
        self._frames_fed += block.size // self.channels
        self._feed(block)

    def _feed_wav(self):
        """
        Replay the WAV in real time, paced by the clock rather than by ticks.

        Counting a fixed number of frames per timer callback silently replays
        slow: a 20 ms ROS timer plus callback overhead lands nearer 20.6 ms, so
        the file drifts ~3% behind and window start indices stop falling on a
        predictable grid. Deriving the target from elapsed time removes the drift
        regardless of how punctual the timer is.
        """
        if self._eof:
            return
        now = self.get_clock().now()
        if self._feed_started_at is None:
            self._feed_started_at = now
        elapsed = (now - self._feed_started_at).nanoseconds / 1e9
        target = int(elapsed * self.source_sample_rate)
        self._read_wav(target - self._frames_fed)

    # ---- publishing ----------------------------------------------------

    def _tick(self):
        """Drain the source, then publish one spectrogram if the buffer is full."""
        # A DAQ node that silently stops keeping up is worse than one that says so.
        now = self.get_clock().now()
        if self._last_tick is not None:
            gap = (now - self._last_tick).nanoseconds / 1e9
            if gap > 1.8 / self.publish_rate_hz:
                self._late_ticks += 1
                self.get_logger().warn(
                    f'Tick was {gap * 1000:.0f} ms late '
                    f'(expected {1000.0 / self.publish_rate_hz:.0f} ms); '
                    f'{self._late_ticks} so far.')
        self._last_tick = now

        if self.source == 'mic':
            self._pump_source()
        elif not self.wav_realtime:
            # Deterministic replay: prime the buffer exactly full, then advance
            # exactly one hop per publish. Window start indices are then exact
            # multiples of the hop -- 0, hop, 2*hop -- which is what reproducible
            # offline processing and cross-pipeline comparison need. Real-time
            # replay cannot promise that, because the indices depend on when the
            # timers actually fire.
            self._read_wav(self.buffer_frames if self._frames_fed == 0
                           else self._hop_frames)

        with self._lock:
            needed = self.buffer_frames * self.channels
            if self._ring.size < needed or self._anchor_ns is None:
                if self._eof and self.shutdown_when_done:
                    self._finish()
                return
            block = self._ring[-needed:].copy()
            start_index = self._frames_written - self.buffer_frames
            anchor_ns = self._anchor_ns
            anchor_index = self._anchor_index

        # No new audio since the last tick means the same buffer would be sent
        # twice under two different timestamps. Skip instead: a consumer must be
        # able to trust that a new message means new sound.
        if start_index == self._last_index:
            if self._eof and self.shutdown_when_done:
                self._finish()
            return
        self._last_index = start_index

        fbank, gain = buffer_to_fbank(block, self.channels, self.source_sample_rate,
                                      self.config)
        offset_ns = int(round((start_index - anchor_index) / self.source_sample_rate
                              * 1e9))
        stamp_ns = anchor_ns + offset_ns
        stamp = Time(nanoseconds=stamp_ns,
                     clock_type=self.get_clock().clock_type).to_msg()

        # A window that starts in the future is a clock error, not something to
        # clamp away: clamping would hide exactly the defect this reports.
        age_ms = (self.get_clock().now().nanoseconds - stamp_ns) / 1e6
        if age_ms < 0.0:
            self._future_stamps += 1
            self.get_logger().warn(
                f'Window start is {-age_ms:.0f} ms in the FUTURE - the sample '
                f'clock and the ROS clock disagree. Not clamping; characterise '
                f'the timing before trusting these stamps.',
                throttle_duration_sec=5.0)

        msg = Spectrogram()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.start_sample_index = int(start_index)
        msg.source_sample_rate = int(self.source_sample_rate)
        msg.analysis_sample_rate = int(self.config.analysis_sample_rate)
        msg.window_duration_sec = float(self.buffer_sec)
        msg.frame_length_ms = float(self.config.frame_length_ms)
        msg.frame_shift_ms = float(self.config.frame_shift_ms)
        msg.peak_gain = float(gain)
        msg.n_frames = int(fbank.shape[0])
        msg.n_mels = int(fbank.shape[1])
        msg.feature_type = FEATURE_TYPE
        msg.preprocessing_id = self.config.preprocessing_id
        # Assign the numpy buffer directly. tolist() would allocate n_frames *
        # n_mels Python floats per message -- 188k objects/s at 15 Hz -- which
        # drives the generational collector into multi-second gen-2 pauses.
        msg.data = np.ascontiguousarray(fbank.reshape(-1), dtype=np.float32)
        self.publisher.publish(msg)

        if self.image_publisher is not None:
            self.image_publisher.publish(self._to_image(fbank, stamp))

        if not self._logged_first:
            self._logged_first = True
            self.get_logger().info(
                f'First spectrogram: {msg.n_frames} x {msg.n_mels} (time-major), '
                f'range [{fbank.min():.2f}, {fbank.max():.2f}], peak_gain={gain:.5f}, '
                f'start_index={msg.start_sample_index}')

        if self._eof and self.shutdown_when_done:
            self._finish()

    def _to_image(self, fbank, stamp):
        """Render a filterbank as a colour-mapped image with low mel bins at the bottom."""
        scaled = (fbank - self.image_min) / (self.image_max - self.image_min)
        indexed = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
        # (frames, mels) -> (mels, frames), then flip so low frequencies sit low.
        rgb = self._lut[indexed.T[::-1]]

        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = self.frame_id
        img.height, img.width = rgb.shape[0], rgb.shape[1]
        img.encoding = 'rgb8'
        img.is_bigendian = 0
        img.step = img.width * 3
        img.data = rgb.reshape(-1).tobytes()
        return img

    def _finish(self):
        """Stop the timer once the WAV source is exhausted."""
        self.timer.cancel()
        rclpy.shutdown()

    def destroy_node(self):
        """Close the audio stream or WAV file before shutting down."""
        if (self._future_stamps or self._adc_time_missing or self._status_events
                or self._timing_discontinuities):
            self.get_logger().warn(
                f'Timing summary: {self._future_stamps} future window stamps, '
                f'{self._adc_time_missing} callbacks without ADC time, '
                f'{self._status_events} PortAudio status events, '
                f'{self._timing_discontinuities} clock disagreements '
                f'(worst {self._worst_residual_ms:.0f} ms).')
        if self._timing_log is not None:
            self._timing_log.close()
            self._timing_log = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._wav is not None:
            self._wav.close()
            self._wav = None
        super().destroy_node()


def main(args=None):
    """Run the audio_daq node until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = AudioDaqNode()
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
