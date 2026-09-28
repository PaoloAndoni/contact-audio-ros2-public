"""
Publish a fake robot state whose pose is a known function of absolute time.

Exists so the audio/robot alignment can be developed and VERIFIED without the
lab. The motion is deliberately analytic:

    position_i(t) = amplitude_i * sin(2*pi*freq_i*t + phase_i)

with t the absolute ROS time in seconds. Because it depends only on absolute
time -- not on when the node started -- anything that knows a timestamp can
recompute exactly where the arm was, with no state to pass around. That is what
lets verify_alignment.py check the aligned dataset against ground truth instead
of merely checking it looks plausible.

Frequencies are mutually prime-ish and distinct per joint, so an alignment that
is offset in time cannot accidentally match on all six.

Usage: python3 scripts/fake_robot_state.py [--rate 100]
"""
import argparse
import math

from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# amplitude (rad), frequency (Hz), phase (rad) -- the ground truth
MOTION = [
    (0.80, 0.070, 0.0),
    (0.60, 0.110, 0.7),
    (1.10, 0.130, 1.4),
    (0.45, 0.190, 2.1),
    (0.90, 0.230, 2.8),
    (0.30, 0.290, 3.5),
]

BASE_FRAME = 'base_link'
TOOL_FRAME = 'tool0'


def joint_positions(t_seconds):
    """Return the exact joint positions at an absolute time, in seconds."""
    return [a * math.sin(2.0 * math.pi * f * t_seconds + p) for a, f, p in MOTION]


def tool_translation(t_seconds):
    """Return a simple end-effector translation derived from the same clock."""
    q = joint_positions(t_seconds)
    return (0.50 + 0.10 * q[0], 0.20 * q[1], 0.35 + 0.08 * q[2])


# --- tool ORIENTATION, added for action derivation ---------------------------
# This node used to broadcast a constant identity quaternion. That is fine for
# checking joint alignment, but it makes the ROTATION half of a derived action
# identically zero -- so the six rotation numbers of a 10-D action would be
# verified against nothing, and an off-by-one in the step index could not
# possibly show up in them.
#
# joint_positions() and tool_translation() are UNCHANGED. verify_alignment.py
# compares joint positions and is unaffected: it measures exactly what it
# measured before. Only the previously-constant rotation field gains content.
#
# Driven by the three WRIST joints (0.19, 0.23, 0.29 Hz), which are distinct
# from each other and from the joints driving the translation, so a shifted
# alignment cannot accidentally match on both halves of the action.
ROLL_GAIN, PITCH_GAIN, YAW_GAIN = 0.30, 0.25, 0.40


def tool_rpy(t_seconds):
    """Return the exact tool roll, pitch and yaw at an absolute time."""
    q = joint_positions(t_seconds)
    return (ROLL_GAIN * q[3], PITCH_GAIN * q[4], YAW_GAIN * q[5])


def tool_rotation(t_seconds):
    """
    Return the tool orientation as an XYZW quaternion, scalar LAST.

    XYZW because that is geometry_msgs/Quaternion's own order, so the fields
    can be assigned straight across. The RPY convention is the ROS standard
    (intrinsic Z-Y-X, i.e. yaw then pitch then roll).
    """
    roll, pitch, yaw = tool_rpy(t_seconds)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def tool_pose(t_seconds):
    """Return the exact end-effector pose [x, y, z, qx, qy, qz, qw]."""
    return tuple(tool_translation(t_seconds)) + tool_rotation(t_seconds)


class FakeRobotState(Node):
    """Publish /joint_states and /tf following the analytic motion above."""

    def __init__(self, rate_hz):
        """Set up publishers and the timer."""
        super().__init__('fake_robot_state')
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.publisher = self.create_publisher(JointState, '/joint_states', qos)
        self.broadcaster = TransformBroadcaster(self)
        self.count = 0
        self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(
            f'Publishing /joint_states and /tf at {rate_hz:g} Hz. Motion is a pure '
            f'function of absolute time, so alignment can be checked exactly.')
        for name, (a, f, p) in zip(JOINT_NAMES, MOTION):
            self.get_logger().info(f'  {name}: {a:.2f} rad @ {f:.3f} Hz, phase {p:.2f}')
        self.get_logger().info(
            f'  tool orientation: roll/pitch/yaw = {ROLL_GAIN:g}/{PITCH_GAIN:g}/'
            f'{YAW_GAIN:g} x the three wrist joints')

    def tick(self):
        """Publish one joint state and one transform."""
        now = self.get_clock().now()
        t = now.nanoseconds / 1e9
        stamp = now.to_msg()

        msg = JointState()
        msg.header.stamp = stamp
        msg.name = JOINT_NAMES
        msg.position = joint_positions(t)
        self.publisher.publish(msg)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = BASE_FRAME
        tf.child_frame_id = TOOL_FRAME
        x, y, z = tool_translation(t)
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        qx, qy, qz, qw = tool_rotation(t)
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.broadcaster.sendTransform(tf)

        self.count += 1
        if self.count % 500 == 0:
            self.get_logger().info(f'{self.count} states published')


def main():
    """Run the fake robot until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rate', type=float, default=100.0)
    args = parser.parse_args()

    rclpy.init()
    node = FakeRobotState(args.rate)
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
