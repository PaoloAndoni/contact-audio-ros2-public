#!/usr/bin/env python3
"""
observation_node - prove the policy input path assembles live, and say what it costs.

    /audio/embedding  (reference clock)  -+
    <camera>/image_raw                    +-> ObservationAssembler -> report / NPZ
    /tf  base -> tool                    -+

It publishes nothing. There is no ROS message for an observation that would not
be worse than the arrays themselves -- a few hundred kilobytes of images per
tick, serialised, so a policy in the same process could deserialise them again.
What is genuinely needed is proof that the three streams can be assembled at
the audio stamp on a live graph, and a measurement of how often they cannot.
That is this node's whole job, and `dump_path` writes what it assembled so the
live result can be compared against the offline one.

POSE SOURCE: tf2 is polled on a fixed timer and every distinct transform is fed
into the assembler's own buffer, rather than calling lookup_transform() at the
reference instant. tf2 would interpolate too, but with its own rules -- so the
observations a policy sees live would be built by different maths than the ones
it was trained on offline. One interpolation policy, in one place, used by both.
"""
from contact_audio_interfaces.msg import AudioEmbedding
from contact_audio_perception.observation import (
    failure_category,
    image_to_array,
    ObservationAssembler,
    ObservationIncomplete,
    ObservationSpec,
    stamp_to_ns,
    transform_to_pose,
)
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener


class ObservationNode(Node):
    """Assemble observations at every audio stamp and report what happened."""

    def __init__(self):
        """Declare parameters, build buffers and subscribe."""
        super().__init__('observation_node')

        self.declare_parameter('audio_topic', '/audio/embedding')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'tool0')
        self.declare_parameter('n_images', 2)
        self.declare_parameter('audio_history_sec', 2.0)
        self.declare_parameter('audio_rate_hz', 15.0)
        self.declare_parameter('pose_dt_sec', 0.0)      # 0 = one audio step
        self.declare_parameter('max_image_age_sec', 0.5)
        self.declare_parameter('max_pose_gap_sec', 0.05)
        self.declare_parameter('tf_poll_hz', 100.0)
        self.declare_parameter('image_rate_hz', 30.0)
        # The reference is the audio buffer's FIRST sample, so it trails
        # wall-clock by the buffer length plus inference. Buffers are sized from
        # this; too small and nothing old enough to use is ever held.
        self.declare_parameter('max_reference_lag_sec', 2.0)
        self.declare_parameter('stats_period_sec', 5.0)
        self.declare_parameter('dump_path', '')
        self.declare_parameter('dump_limit', 200)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.dump_path = self.get_parameter('dump_path').value
        self.dump_limit = int(self.get_parameter('dump_limit').value)

        # The spec needs the embedding width and the image size, and asking the
        # user to repeat numbers the messages already carry is how a config
        # drifts out of step with the encoder it describes. Built on first sight
        # of each, then frozen.
        self._audio_dim = None
        self._image_shape = None
        self.assembler = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_tf_ns = -1

        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.on_image, sensor_qos)
        self.create_subscription(
            AudioEmbedding, self.get_parameter('audio_topic').value,
            self.on_audio, sensor_qos)

        tf_hz = float(self.get_parameter('tf_poll_hz').value)
        self.create_timer(1.0 / tf_hz, self.poll_tf)

        self._assembled = 0
        self._failed = 0
        self._reasons = {}
        self._pose_gaps = []
        self._image_ages = []
        self._dumped = []
        stats_period = float(self.get_parameter('stats_period_sec').value)
        if stats_period > 0.0:
            self.create_timer(stats_period, self.log_stats)

        self.get_logger().info(
            f'Assembling at every {self.get_parameter("audio_topic").value} stamp, '
            f'from {self.get_parameter("image_topic").value} and '
            f'{self.base_frame} -> {self.tool_frame}')

    def _build_assembler(self):
        """Create the assembler once both stream shapes are known."""
        pose_dt = float(self.get_parameter('pose_dt_sec').value)
        rate = float(self.get_parameter('audio_rate_hz').value)
        spec = ObservationSpec(
            n_images=int(self.get_parameter('n_images').value),
            image_shape=self._image_shape,
            audio_dim=self._audio_dim,
            audio_history_sec=float(self.get_parameter('audio_history_sec').value),
            audio_rate_hz=rate,
            audio_span_tolerance_sec=1.5 / rate,
            pose_dt_sec=pose_dt if pose_dt > 0.0 else 1.0 / rate,
            max_image_age_sec=float(self.get_parameter('max_image_age_sec').value),
            max_pose_gap_sec=float(self.get_parameter('max_pose_gap_sec').value),
            image_rate_hz=float(self.get_parameter('image_rate_hz').value),
            pose_rate_hz=float(self.get_parameter('tf_poll_hz').value),
            max_reference_lag_sec=float(
                self.get_parameter('max_reference_lag_sec').value),
        )
        self.assembler = ObservationAssembler(spec)
        self.get_logger().info(f'Observation shapes: {spec.shapes()}')
        self.get_logger().info(
            f'Buffers hold {spec.buffer_sec:g} s: images '
            f'{self.assembler.images.capacity}, audio {self.assembler.audio.capacity}, '
            f'poses {self.assembler.poses.capacity}')
        self.get_logger().info(
            f'Audio history {spec.audio_history_sec:g} s = {spec.n_audio} items at '
            f'{spec.audio_rate_hz:g} Hz; pose pair spans {spec.pose_dt_sec * 1e3:.1f} ms')

    def _ensure_assembler(self):
        """Build the assembler as soon as both shapes have been observed."""
        if self.assembler is None and self._audio_dim and self._image_shape:
            self._build_assembler()

    def poll_tf(self):
        """Feed every newly available base->tool transform into the pose buffer."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.tool_frame, rclpy.time.Time())
        except Exception as exc:
            self.get_logger().warn(f'no {self.base_frame} -> {self.tool_frame}: {exc}',
                                   throttle_duration_sec=5.0)
            return
        stamp = stamp_to_ns(transform.header)
        if stamp <= self._last_tf_ns:
            return          # polling faster than the broadcaster; not a new sample
        if self.assembler is None:
            # Not yet built. Leave _last_tf_ns alone so this sample is picked up
            # on the next poll instead of being skipped as "already seen".
            return
        self._last_tf_ns = stamp
        self.assembler.add_pose(stamp, transform_to_pose(transform.transform))

    def on_image(self, msg):
        """Buffer one camera frame."""
        try:
            array = image_to_array(msg)
        except ValueError as exc:
            self.get_logger().warn(f'dropping image: {exc}', throttle_duration_sec=5.0)
            return
        if self._image_shape is None:
            self._image_shape = array.shape
            self._ensure_assembler()
        if self.assembler is None or array.shape != self.assembler.spec.image_shape:
            return
        self.assembler.add_image(stamp_to_ns(msg.header), array)

    def on_audio(self, msg):
        """Buffer one embedding, then try to assemble an observation at its stamp."""
        vector = np.asarray(msg.embedding, dtype=np.float32)
        if vector.size != int(msg.embedding_dim):
            self.get_logger().warn(
                f'embedding declares {msg.embedding_dim} values but carries '
                f'{vector.size}; dropping', throttle_duration_sec=5.0)
            return
        if self._audio_dim is None:
            self._audio_dim = int(vector.size)
            self._ensure_assembler()
        if self.assembler is None or vector.size != self.assembler.spec.audio_dim:
            return

        reference = stamp_to_ns(msg.header)
        self.assembler.add_audio(reference, vector)
        try:
            observation = self.assembler.assemble(reference)
        except ObservationIncomplete as exc:
            self._failed += 1
            reason = failure_category(exc)
            self._reasons[reason] = self._reasons.get(reason, 0) + 1
            self.get_logger().warn(f'could not assemble: {exc}',
                                   throttle_duration_sec=5.0)
            return

        self._assembled += 1
        self._pose_gaps.append(observation.pose_gap_ns / 1e6)
        self._image_ages.append(observation.image_age_sec * 1e3)
        if self.dump_path and len(self._dumped) < self.dump_limit:
            self._dumped.append(observation)
        if self._assembled == 1:
            self.get_logger().info(
                f'First observation at {reference}: images {observation.images.shape}, '
                f'audio {observation.audio.shape}, pose gap '
                f'{observation.pose_gap_ns / 1e6:.2f} ms, newest frame '
                f'{observation.image_age_sec * 1e3:.1f} ms old')

    def log_stats(self):
        """Report the assembly success rate and the staleness actually observed."""
        total = self._assembled + self._failed
        if total == 0:
            return
        if self._pose_gaps:
            detail = (f'pose gap median {np.median(self._pose_gaps):.2f} ms, '
                      f'newest frame median {np.median(self._image_ages):.1f} ms old')
        else:
            detail = 'nothing assembled yet'
        self.get_logger().info(
            f'{self._assembled}/{total} assembled | {detail}')
        for reason, count in sorted(self._reasons.items(), key=lambda kv: -kv[1])[:3]:
            self.get_logger().info(f'   {count} x {reason}')
        self._pose_gaps.clear()
        self._image_ages.clear()

    def write_dump(self):
        """Write the assembled observations to an NPZ, if one was requested."""
        if not self.dump_path or not self._dumped:
            return
        np.savez(
            self.dump_path,
            reference_stamp_ns=np.array(
                [o.reference_stamp_ns for o in self._dumped], dtype=np.int64),
            images=np.stack([o.images for o in self._dumped]),
            image_stamps_ns=np.stack([o.image_stamps_ns for o in self._dumped]),
            audio=np.stack([o.audio for o in self._dumped]),
            audio_stamps_ns=np.stack([o.audio_stamps_ns for o in self._dumped]),
            pose_current=np.stack([o.pose_current for o in self._dumped]),
            pose_previous=np.stack([o.pose_previous for o in self._dumped]),
            pose_dt_sec=np.array(self._dumped[0].pose_dt_sec),
            pose_gap_ns=np.array([o.pose_gap_ns for o in self._dumped], dtype=np.int64),
        )
        self.get_logger().info(
            f'wrote {len(self._dumped)} observations to {self.dump_path}')


def main(args=None):
    """Run the observation node until shutdown, then flush any dump."""
    rclpy.init(args=args)
    node = None
    try:
        node = ObservationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.write_dump()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
