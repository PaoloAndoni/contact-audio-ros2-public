"""
Turn a sequence of end-effector poses into the action chunks a policy predicts.

REPRESENTATION: 10 numbers per step, matching ManiWAV (which follows UMI):

    [0:3]  translation delta
    [3:9]  relative rotation, 6-D continuous representation (Zhou et al. 2019)
    [9]    gripper command, ABSOLUTE

and a chunk of 16 future steps per observation. Matched rather than invented
because it is the representation the published contact-audio result was
obtained with; deviating would make our numbers incomparable with theirs for no
gain. The two parts that are NOT arbitrary: rotation must be continuous over
SO(3) or a regression policy cannot fit it (quaternions and Euler angles both
have discontinuities a network will smear across), and the gripper must be
absolute, since differencing a width makes a policy integrate its own error.

RELATIVE, NOT ABSOLUTE, and that is what makes this work for a collection
method we do not yet know. A hand-guided arm, a teleoperated arm and a handheld
gripper all produce a pose trajectory in some base frame, but they do not
produce the SAME base frame, and a handheld device has no joint states at all.
Deltas in the end-effector's own frame (`frame='ee'`) are invariant to where
that base sits, so the same derivation applies unchanged to all three. Nothing
in this module refers to joints, controllers or a command topic.

THE STEP INDEX -- the thing that is easy to get wrong and impossible to notice.
Steps are indexed k = 0 .. K-1, step k anchored at reference time t_k (an audio
window's header.stamp). Observation_k is assembled at t_k. Then

    action_k[j] = delta from pose(t_{k+j}) to pose(t_{k+j+1}),  j = 0 .. L-1

so action_k[0] is the motion from NOW to the NEXT step: the first thing the
policy executes, never the motion that has already happened. Shifting this by
one produces actions that are smooth, plausible and wrong, which is why
verify_actions.py compares against a closed form instead of eyeballing them.

Sample k therefore needs poses up to index k+L, and is valid iff k + L <= K-1.
"""
from contact_audio_perception.pose_math import pose_delta_10d
import numpy as np

ACTION_DIM = 10
DEFAULT_CHUNK_LENGTH = 16
DEFAULT_FRAME = 'ee'

# Slice names, so a consumer never has to hardcode the layout.
TRANSLATION = slice(0, 3)
ROTATION_6D = slice(3, 9)
GRIPPER = 9


class ActionChunker:
    """Derive fixed-length chunks of relative end-effector actions."""

    def __init__(self, chunk_length=DEFAULT_CHUNK_LENGTH, frame=DEFAULT_FRAME):
        """Fix the chunk length and the frame every delta is expressed in."""
        if chunk_length < 1:
            raise ValueError(f'chunk_length must be >= 1, got {chunk_length}')
        if frame not in ('ee', 'base'):
            raise ValueError(f"frame must be 'ee' or 'base', got {frame!r}")
        self.chunk_length = int(chunk_length)
        self.frame = frame

    def valid_samples(self, n_steps):
        """Return how many chunks a K-step trajectory yields (0 if too short)."""
        return max(0, int(n_steps) - self.chunk_length)

    def chunk(self, poses, grippers, index):
        """
        Return the (chunk_length, 10) action chunk anchored at step `index`.

        `poses` is (K, 7) and `grippers` is (K,); row j of the result is the
        move from step index+j to step index+j+1, carrying the gripper command
        of the TARGET step index+j+1.
        """
        poses = np.asarray(poses, dtype=np.float64)
        grippers = np.asarray(grippers, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError(f'poses must be (K, 7), got {poses.shape}')
        if grippers.shape != (poses.shape[0],):
            raise ValueError(
                f'grippers must be ({poses.shape[0]},), got {grippers.shape}')
        last = index + self.chunk_length
        if index < 0 or last >= poses.shape[0]:
            raise IndexError(
                f'chunk at step {index} needs poses up to {last}, but only '
                f'{poses.shape[0]} steps exist. Valid anchors are 0..'
                f'{self.valid_samples(poses.shape[0]) - 1}.')
        out = np.empty((self.chunk_length, ACTION_DIM), dtype=np.float64)
        for j in range(self.chunk_length):
            out[j] = pose_delta_10d(
                poses[index + j], poses[index + j + 1],
                gripper=grippers[index + j + 1], frame=self.frame)
        return out

    def all_chunks(self, poses, grippers):
        """Return every valid chunk as (n_samples, chunk_length, 10)."""
        poses = np.asarray(poses, dtype=np.float64)
        count = self.valid_samples(poses.shape[0])
        if count == 0:
            return np.empty((0, self.chunk_length, ACTION_DIM), dtype=np.float64)
        return np.stack([self.chunk(poses, grippers, k) for k in range(count)])


def channels_grew(shifted, aligned, factor):
    """
    Say whether every channel grew past `factor` times its own aligned error.

    Channel errors are NOT comparable across channels: TRANSLATION is in metres
    and ROTATION_6D is dimensionless. A check that takes min() or max() across
    the two and compares the result against the other is comparing a length to
    a bare number, so its verdict turns on the arbitrary ratio between the
    units rather than on the alignment. Hold each channel to its own baseline.

    all(), not any(): a deliberately corrupted alignment has to move every
    channel it touches. One channel growing while another stands still means
    the corruption went undetected in that channel.

    Pass only channels that carry signal. The gripper is a constant zero
    whenever nothing drives it, and a channel that cannot move can never grow.
    """
    if len(shifted) != len(aligned):
        raise ValueError('shifted and aligned must describe the same channels')
    if not shifted:
        return False
    return all(s > a * factor for s, a in zip(shifted, aligned))


def describe_layout():
    """Return a human-readable description of the 10 action channels."""
    return {
        'translation': 'action[0:3]  delta position, metres, in the chosen frame',
        'rotation_6d': 'action[3:9]  first two columns of the relative rotation matrix',
        'gripper': 'action[9]    absolute gripper command of the target step',
    }
