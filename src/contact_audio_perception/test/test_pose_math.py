"""Tests for the rigid-body maths the observation and action paths share."""
import math

from contact_audio_perception.pose_math import (
    matrix_to_quaternion,
    pose_delta_10d,
    pose_interpolate,
    quaternion_canonical,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_to_matrix,
    relative_rotation,
    relative_translation,
    rotation_from_6d,
    rotation_to_6d,
)
import numpy as np
import pytest


def _quat_rpy(roll, pitch, yaw):
    """Build an XYZW quaternion from roll-pitch-yaw, independently of the module."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def _random_quats(count, seed=0):
    """Return `count` unit quaternions spread over the whole rotation group."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(count):
        q = rng.randn(4)
        out.append(q / np.linalg.norm(q))
    return out


# ---- quaternions ----

def test_quaternion_to_matrix_is_a_rotation():
    """Every converted quaternion gives an orthonormal matrix with det +1."""
    for q in _random_quats(20):
        m = quaternion_to_matrix(q)
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(m), 1.0, atol=1e-12)


def test_matrix_to_quaternion_round_trips_including_180_degrees():
    """
    The trace-only formula loses precision near a half turn; Shepperd's does not.

    A 180-degree rotation has trace -1, so the naive 0.5/sqrt(trace+1) divides
    by zero there. These are exactly the cases a wrist can reach.
    """
    angles = [0.0, 1e-9, math.pi / 2, math.pi - 1e-7, math.pi]
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
                 np.array([1.0, 1.0, 1.0]) / math.sqrt(3)):
        for angle in angles:
            q = np.concatenate([axis * math.sin(angle / 2), [math.cos(angle / 2)]])
            back = matrix_to_quaternion(quaternion_to_matrix(q))
            assert np.allclose(quaternion_to_matrix(back), quaternion_to_matrix(q),
                               atol=1e-7)


def test_canonical_form_collapses_the_double_cover():
    """Sign-flipped quaternions are one rotation, so both canonicalise alike."""
    for q in _random_quats(10, seed=3):
        assert np.allclose(quaternion_canonical(q), quaternion_canonical(-q))
        assert quaternion_canonical(q)[3] >= 0.0


def test_quaternion_multiply_matches_matrix_multiply():
    """The Hamilton product must agree with composing the rotation matrices."""
    a, b = _random_quats(2, seed=7)
    composed = quaternion_to_matrix(quaternion_multiply(a, b))
    assert np.allclose(composed, quaternion_to_matrix(a) @ quaternion_to_matrix(b),
                       atol=1e-12)


def test_quaternion_inverse_undoes_the_rotation():
    """Composing a quaternion with its inverse gives the identity."""
    for q in _random_quats(10, seed=11):
        product = quaternion_multiply(q, quaternion_inverse(q))
        assert np.allclose(quaternion_to_matrix(product), np.eye(3), atol=1e-12)


def test_zero_quaternion_is_refused():
    """A quaternion with no direction is a bug upstream, not a rotation."""
    with pytest.raises(ValueError, match='no direction'):
        quaternion_to_matrix([0.0, 0.0, 0.0, 0.0])


# ---- slerp ----

def test_slerp_endpoints_and_midpoint_angle():
    """Slerp hits both ends exactly and halves the angle at the midpoint."""
    a = _quat_rpy(0, 0, 0)
    b = _quat_rpy(0, 0, 1.2)
    assert np.allclose(quaternion_slerp(a, b, 0.0), a, atol=1e-12)
    assert np.allclose(quaternion_canonical(quaternion_slerp(a, b, 1.0)),
                       quaternion_canonical(b), atol=1e-12)
    mid = quaternion_slerp(a, b, 0.5)
    assert np.allclose(quaternion_canonical(mid),
                       quaternion_canonical(_quat_rpy(0, 0, 0.6)), atol=1e-12)


def test_slerp_moves_at_a_constant_angular_rate_but_lerp_does_not():
    """
    The reason slerp is used at all, stated as a number.

    Renormalised linear interpolation is off by degrees at the quarter points
    for a wide rotation -- a bias on every pose sampled between two messages.
    """
    a = _quat_rpy(0, 0, 0.0)
    b = _quat_rpy(0, 0, 2.4)
    for fraction in (0.25, 0.5, 0.75):
        exact = _quat_rpy(0, 0, 2.4 * fraction)
        slerped = quaternion_slerp(a, b, fraction)
        lerped = (1 - fraction) * a + fraction * b
        lerped = lerped / np.linalg.norm(lerped)
        slerp_err = np.abs(quaternion_canonical(slerped) - quaternion_canonical(exact))
        lerp_err = np.abs(quaternion_canonical(lerped) - quaternion_canonical(exact))
        assert slerp_err.max() < 1e-12
        if fraction != 0.5:
            assert lerp_err.max() > 1e-3


def test_slerp_takes_the_shortest_arc_across_a_sign_flip():
    """A sign-flipped sample must not turn a small rotation into a huge one."""
    a = _quat_rpy(0, 0, 0.1)
    b = _quat_rpy(0, 0, 0.3)
    straight = quaternion_slerp(a, b, 0.5)
    flipped = quaternion_slerp(a, -b, 0.5)
    assert np.allclose(quaternion_to_matrix(straight), quaternion_to_matrix(flipped),
                       atol=1e-12)


def test_pose_interpolate_is_linear_in_translation():
    """Translation interpolates linearly while rotation slerps."""
    a = np.concatenate([[0.0, 0.0, 0.0], _quat_rpy(0, 0, 0)])
    b = np.concatenate([[1.0, 2.0, 3.0], _quat_rpy(0, 0, 1.0)])
    mid = pose_interpolate(a, b, 0.25)
    assert np.allclose(mid[:3], [0.25, 0.5, 0.75])
    assert np.allclose(quaternion_canonical(mid[3:]),
                       quaternion_canonical(_quat_rpy(0, 0, 0.25)), atol=1e-12)


# ---- 6-D rotation ----

def test_6d_rotation_round_trips_exactly():
    """rotation_from_6d inverts rotation_to_6d for any real rotation."""
    for q in _random_quats(25, seed=5):
        m = quaternion_to_matrix(q)
        assert np.allclose(rotation_from_6d(rotation_to_6d(m)), m, atol=1e-12)


def test_6d_layout_is_the_first_two_columns():
    """The layout is pinned: a transposed copy would still round-trip."""
    m = quaternion_to_matrix(_quat_rpy(0.3, -0.2, 0.9))
    six = rotation_to_6d(m)
    assert np.allclose(six[:3], m[:, 0])
    assert np.allclose(six[3:], m[:, 1])


def test_6d_projects_a_noisy_vector_back_onto_a_rotation():
    """A network's not-quite-orthonormal output still yields a valid rotation."""
    rng = np.random.RandomState(0)
    noisy = rotation_to_6d(quaternion_to_matrix(_quat_rpy(0.1, 0.2, 0.3)))
    noisy = noisy + rng.randn(6) * 0.05
    m = rotation_from_6d(noisy)
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(m), 1.0, atol=1e-12)


def test_6d_is_continuous_where_euler_angles_are_not():
    """
    Two nearly identical rotations must give nearly identical numbers.

    This is the whole reason the representation exists: yaw crossing pi jumps
    by 2*pi as an Euler angle, and a regression policy cannot fit that.
    """
    below = rotation_to_6d(quaternion_to_matrix(_quat_rpy(0, 0, math.pi - 1e-6)))
    above = rotation_to_6d(quaternion_to_matrix(_quat_rpy(0, 0, -math.pi + 1e-6)))
    assert np.abs(below - above).max() < 1e-5


# ---- relative poses ----

def test_relative_rotation_in_ee_frame_is_base_invariant():
    """
    Re-mounting the robot must not change an ee-frame action.

    This is what makes one derivation work for hand-guiding, teleoperation and
    a handheld device, which do not share a base frame.
    """
    rotate_base = quaternion_to_matrix(_quat_rpy(0.4, -0.3, 1.1))
    a, b = _random_quats(2, seed=13)
    direct = relative_rotation(a, b, frame='ee')
    moved = relative_rotation(
        matrix_to_quaternion(rotate_base @ quaternion_to_matrix(a)),
        matrix_to_quaternion(rotate_base @ quaternion_to_matrix(b)), frame='ee')
    assert np.allclose(direct, moved, atol=1e-12)


def test_relative_rotation_in_base_frame_is_not_base_invariant():
    """The control: base-frame actions DO change when the base moves."""
    rotate_base = quaternion_to_matrix(_quat_rpy(0.4, -0.3, 1.1))
    a, b = _random_quats(2, seed=13)
    direct = relative_rotation(a, b, frame='base')
    moved = relative_rotation(
        matrix_to_quaternion(rotate_base @ quaternion_to_matrix(a)),
        matrix_to_quaternion(rotate_base @ quaternion_to_matrix(b)), frame='base')
    assert not np.allclose(direct, moved, atol=1e-3)


def test_relative_translation_frames_differ_by_the_start_rotation():
    """The ee-frame delta is the base-frame delta rotated into the start pose."""
    q = _quat_rpy(0, 0, math.pi / 2)
    start = np.concatenate([[1.0, 1.0, 1.0], q])
    end = np.concatenate([[2.0, 1.0, 1.0], q])
    assert np.allclose(relative_translation(start, end, frame='base'), [1, 0, 0])
    # Yawed 90 degrees, a +x move in the base frame is a -y move in the tool's.
    assert np.allclose(relative_translation(start, end, frame='ee'), [0, -1, 0],
                       atol=1e-12)


def test_pose_delta_of_a_pose_with_itself_is_the_identity_action():
    """Standing still means zero translation and an identity 6-D rotation."""
    pose = np.concatenate([[0.3, -0.2, 0.9], _quat_rpy(0.2, 0.3, -0.4)])
    action = pose_delta_10d(pose, pose, gripper=0.5)
    assert np.allclose(action[:3], 0.0, atol=1e-12)
    assert np.allclose(rotation_from_6d(action[3:9]), np.eye(3), atol=1e-12)
    assert action[9] == 0.5


def test_pose_delta_gripper_is_absolute_not_differenced():
    """The gripper channel carries the target's value, whatever the start was."""
    a = np.concatenate([[0.0, 0, 0], _quat_rpy(0, 0, 0)])
    b = np.concatenate([[0.1, 0, 0], _quat_rpy(0, 0, 0)])
    assert pose_delta_10d(a, b, gripper=0.8)[9] == 0.8


def test_unknown_frame_is_refused():
    """A typo'd frame name must fail loudly, not fall through to a default."""
    pose = np.concatenate([[0.0, 0, 0], _quat_rpy(0, 0, 0)])
    with pytest.raises(ValueError, match="must be 'ee' or 'base'"):
        pose_delta_10d(pose, pose, frame='world')
