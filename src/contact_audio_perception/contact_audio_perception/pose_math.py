"""
Rigid-body pose maths shared by the observation assembler and action derivation.

ROS-free and numpy-only on purpose: every quantity here is checked by unit
tests that need neither a robot nor a running graph.

POSE CONVENTION, used everywhere in this package: a pose is a 7-vector

    [x, y, z, qx, qy, qz, qw]

i.e. translation first, then a quaternion in ROS's XYZW order with the scalar
LAST. geometry_msgs/Quaternion also stores x, y, z, w, so a pose can be read
off a TransformStamped field by field with no reordering -- which is exactly
the kind of silent transposition that produces plausible, wrong numbers.

Quaternions are double-cover: q and -q are the same rotation. Anything that
compares, interpolates or differences them here canonicalises the sign first,
otherwise a sign flip between two consecutive samples turns a zero rotation
into a 360-degree one.
"""
import numpy as np

# Below this the two quaternions are parallel to floating-point resolution and
# slerp's sin(theta) denominator is no longer safe; lerp is then exact enough.
SLERP_LINEAR_THRESHOLD = 1.0 - 1e-9


def quaternion_normalise(q):
    """Return the unit quaternion parallel to q, raising if it has no direction."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f'quaternion has no direction: {q}')
    return q / norm


def quaternion_canonical(q):
    """
    Pick the representative of q with a non-negative scalar part.

    q and -q rotate identically, so any comparison between quaternions from
    different sources must fix the sign first.
    """
    q = quaternion_normalise(q)
    return -q if q[3] < 0.0 else q


def quaternion_multiply(a, b):
    """Return the Hamilton product a*b, both in XYZW order."""
    ax, ay, az, aw = np.asarray(a, dtype=np.float64)
    bx, by, bz, bw = np.asarray(b, dtype=np.float64)
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quaternion_inverse(q):
    """Return the inverse of a unit quaternion (its conjugate)."""
    x, y, z, w = quaternion_normalise(q)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quaternion_to_matrix(q):
    """Return the 3x3 rotation matrix of a quaternion in XYZW order."""
    x, y, z, w = quaternion_normalise(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion(matrix):
    """
    Return the XYZW quaternion of a 3x3 rotation matrix.

    Uses Shepperd's method: pick the largest of the four possible divisors, so
    the square root never runs into a near-zero denominator. The naive
    trace-only formula loses all precision near a 180-degree rotation.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f'expected a 3x3 rotation matrix, got {m.shape}')
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s,
                      (m[1, 0] - m[0, 1]) * s, 0.25 / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
                      (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
                      (m[1, 0] - m[0, 1]) / s])
    return quaternion_canonical(q)


def quaternion_slerp(a, b, fraction):
    """
    Interpolate between two rotations along the shortest arc.

    Linear interpolation of quaternion components, then renormalising, is NOT
    the same thing: it moves at a non-constant angular rate, which biases every
    pose sampled between two joint-state messages.
    """
    a = quaternion_normalise(a)
    b = quaternion_normalise(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:          # shortest arc: flip one, they are the same rotation
        b = -b
        dot = -dot
    if dot > SLERP_LINEAR_THRESHOLD:
        return quaternion_normalise(a + fraction * (b - a))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    scale_a = np.sin((1.0 - fraction) * theta) / sin_theta
    scale_b = np.sin(fraction * theta) / sin_theta
    return quaternion_normalise(scale_a * a + scale_b * b)


def pose_interpolate(pose_a, pose_b, fraction):
    """Interpolate a 7-vector pose: linear in translation, slerp in rotation."""
    pose_a = np.asarray(pose_a, dtype=np.float64)
    pose_b = np.asarray(pose_b, dtype=np.float64)
    out = np.empty(7, dtype=np.float64)
    out[:3] = pose_a[:3] + fraction * (pose_b[:3] - pose_a[:3])
    out[3:] = quaternion_slerp(pose_a[3:], pose_b[3:], fraction)
    return out


def rotation_to_6d(matrix):
    """
    Return the 6-D rotation representation of Zhou et al. (2019).

    The first TWO COLUMNS of the rotation matrix, stacked column by column:
    [m00, m10, m20, m01, m11, m21]. The third column is the cross product of
    the first two, so nothing is lost. Unlike Euler angles or quaternions this
    is continuous over SO(3), which is why regression policies predict it --
    and it is what ManiWAV's 10-number action uses.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f'expected a 3x3 rotation matrix, got {m.shape}')
    return np.concatenate([m[:, 0], m[:, 1]])


def rotation_from_6d(vector):
    """
    Rebuild a rotation matrix from the 6-D representation, by Gram-Schmidt.

    The inverse of rotation_to_6d for any valid rotation, and a projection onto
    the nearest rotation for the not-quite-orthonormal vectors a network emits.
    """
    v = np.asarray(vector, dtype=np.float64).reshape(6)
    a1, a2 = v[:3], v[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def relative_rotation(q_from, q_to, frame='ee'):
    """
    Return the rotation taking q_from to q_to, as a 3x3 matrix.

    frame='ee'   -> R_from^T @ R_to, the rotation expressed in the moving
                    end-effector frame. This is the ManiWAV/UMI convention and
                    is invariant to where the robot's base happens to be, which
                    is what lets a handheld-collected demonstration transfer.
    frame='base' -> R_to @ R_from^T, the rotation expressed in the fixed base
                    frame. Correct for a base-mounted camera, but a demonstration
                    recorded at a different base pose no longer matches.
    """
    r_from = quaternion_to_matrix(q_from)
    r_to = quaternion_to_matrix(q_to)
    if frame == 'ee':
        return r_from.T @ r_to
    if frame == 'base':
        return r_to @ r_from.T
    raise ValueError(f"frame must be 'ee' or 'base', got {frame!r}")


def relative_translation(pose_from, pose_to, frame='ee'):
    """
    Return the translation from one pose to another, in the chosen frame.

    frame='ee' rotates the base-frame difference into the start pose's own
    frame, so 'move 1 cm along my own x' means the same thing wherever the arm
    is standing. frame='base' leaves it in base coordinates.
    """
    pose_from = np.asarray(pose_from, dtype=np.float64)
    pose_to = np.asarray(pose_to, dtype=np.float64)
    delta = pose_to[:3] - pose_from[:3]
    if frame == 'base':
        return delta
    if frame == 'ee':
        return quaternion_to_matrix(pose_from[3:]).T @ delta
    raise ValueError(f"frame must be 'ee' or 'base', got {frame!r}")


def pose_delta_10d(pose_from, pose_to, gripper=0.0, frame='ee'):
    """
    Return the 10-number action of ManiWAV: 3 translation + 6-D rotation + 1 gripper.

    Layout is fixed and must not be permuted:
        [0:3]  translation delta, in the frame named by `frame`
        [3:9]  relative rotation as rotation_to_6d(relative_rotation(...))
        [9]    gripper command for the TARGET step

    The gripper channel is the target's absolute value, not a delta: a gripper
    is commanded to a width, and differencing it would make a policy integrate
    open/close errors forever.
    """
    action = np.empty(10, dtype=np.float64)
    action[:3] = relative_translation(pose_from, pose_to, frame=frame)
    action[3:9] = rotation_to_6d(
        relative_rotation(pose_from[3:], pose_to[3:], frame=frame))
    action[9] = float(gripper)
    return action
