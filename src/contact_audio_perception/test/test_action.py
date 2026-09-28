"""Tests for action chunking, with the off-by-one made explicit."""
import math

from contact_audio_perception.action import (
    ACTION_DIM,
    ActionChunker,
    channels_grew,
    GRIPPER,
    ROTATION_6D,
    TRANSLATION,
)
from contact_audio_perception.pose_math import rotation_from_6d
import numpy as np
import pytest


def _straight_line(n_steps, step=0.01):
    """Poses moving along +x at a constant rate, orientation fixed."""
    poses = np.zeros((n_steps, 7))
    poses[:, 0] = np.arange(n_steps) * step
    poses[:, 6] = 1.0
    return poses


def _yawing(n_steps, per_step=0.05):
    """Poses yawing at a constant rate, position fixed."""
    poses = np.zeros((n_steps, 7))
    angles = np.arange(n_steps) * per_step
    poses[:, 5] = np.sin(angles / 2)
    poses[:, 6] = np.cos(angles / 2)
    return poses


def test_chunk_shape_and_count():
    """A K-step trajectory yields K-L chunks of (L, 10)."""
    chunker = ActionChunker(chunk_length=16)
    poses = _straight_line(40)
    grippers = np.zeros(40)
    assert chunker.valid_samples(40) == 24
    assert chunker.chunk(poses, grippers, 0).shape == (16, ACTION_DIM)
    assert chunker.all_chunks(poses, grippers).shape == (24, 16, ACTION_DIM)


def test_short_trajectory_yields_nothing_rather_than_a_padded_chunk():
    """Padding a chunk would teach a policy to stop moving at the end of a demo."""
    chunker = ActionChunker(chunk_length=16)
    poses = _straight_line(10)
    assert chunker.valid_samples(10) == 0
    assert chunker.all_chunks(poses, np.zeros(10)).shape == (0, 16, ACTION_DIM)


def test_action_zero_is_the_move_from_now_to_the_next_step():
    """
    The index contract, pinned.

    action_k[0] must be the step the policy is about to take. If it were the
    step already taken, a policy would be trained to report the past -- which
    looks perfectly smooth and is useless.
    """
    chunker = ActionChunker(chunk_length=3)
    poses = np.zeros((10, 7))
    poses[:, 6] = 1.0
    poses[:, 0] = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]   # deltas 1,2,3,4,...
    chunk = chunker.chunk(poses, np.zeros(10), 2)
    # Anchored at step 2, the next moves are 3 (2->3), 4 (3->4), 5 (4->5).
    assert np.allclose(chunk[:, 0], [3.0, 4.0, 5.0])


def test_an_off_by_one_anchor_is_smooth_and_wrong():
    """
    The trap this module is written against, demonstrated as a number.

    A chunk anchored one step late is just as smooth as the right one -- the
    second difference is identical -- but every value is displaced. Only a
    comparison against known truth separates them.
    """
    chunker = ActionChunker(chunk_length=3)
    poses = np.zeros((10, 7))
    poses[:, 6] = 1.0
    poses[:, 0] = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
    correct = chunker.chunk(poses, np.zeros(10), 2)[:, 0]
    shifted = chunker.chunk(poses, np.zeros(10), 3)[:, 0]
    assert np.allclose(np.diff(correct, 2), np.diff(shifted, 2))    # equally smooth
    assert np.abs(correct - shifted).max() >= 1.0                   # entirely wrong


def test_translation_channel_is_the_difference_not_the_position():
    """Absolute poses leaking into the action channel is a classic slip."""
    chunker = ActionChunker(chunk_length=4)
    poses = _straight_line(20, step=0.01)
    chunk = chunker.chunk(poses, np.zeros(20), 10)
    assert np.allclose(chunk[:, TRANSLATION][:, 0], 0.01)
    assert not np.allclose(chunk[:, TRANSLATION][:, 0], poses[10:14, 0])


def test_rotation_channel_recovers_the_per_step_rotation():
    """The 6-D channel must decode back to the true relative rotation."""
    per_step = 0.05
    chunker = ActionChunker(chunk_length=5)
    poses = _yawing(20, per_step=per_step)
    chunk = chunker.chunk(poses, np.zeros(20), 7)
    for row in chunk:
        matrix = rotation_from_6d(row[ROTATION_6D])
        angle = math.atan2(matrix[1, 0], matrix[0, 0])
        assert np.isclose(angle, per_step, atol=1e-9)


def test_gripper_channel_takes_the_target_step_value():
    """Row j carries the gripper command of step index+j+1, not index+j."""
    chunker = ActionChunker(chunk_length=3)
    poses = _straight_line(10)
    grippers = np.arange(10, dtype=float)
    chunk = chunker.chunk(poses, grippers, 4)
    assert np.allclose(chunk[:, GRIPPER], [5.0, 6.0, 7.0])


def test_ee_frame_actions_survive_re_mounting_the_robot():
    """
    The property that makes one derivation fit three collection methods.

    The same motion recorded with the base rotated must give identical ee-frame
    actions, because a handheld device and a mounted arm do not agree on a base.
    """
    from contact_audio_perception.pose_math import (
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
    rng = np.random.RandomState(4)
    poses = np.zeros((20, 7))
    for i in range(20):
        q = rng.randn(4)
        poses[i, :3] = rng.randn(3) * 0.1
        poses[i, 3:] = q / np.linalg.norm(q)

    rotate = quaternion_to_matrix([0.1, 0.3, -0.2, 0.9])
    rotate = rotate / np.linalg.norm(rotate[:, 0])
    moved = np.zeros_like(poses)
    for i in range(20):
        moved[i, :3] = rotate @ poses[i, :3] + np.array([5.0, -2.0, 1.0])
        moved[i, 3:] = matrix_to_quaternion(rotate @ quaternion_to_matrix(poses[i, 3:]))

    chunker = ActionChunker(chunk_length=3, frame='ee')
    assert np.allclose(chunker.chunk(poses, np.zeros(20), 5),
                       chunker.chunk(moved, np.zeros(20), 5), atol=1e-9)

    base_chunker = ActionChunker(chunk_length=3, frame='base')
    assert not np.allclose(base_chunker.chunk(poses, np.zeros(20), 5),
                           base_chunker.chunk(moved, np.zeros(20), 5), atol=1e-3)


def test_out_of_range_anchor_names_the_valid_range():
    """Running off the end must say what the valid anchors were."""
    chunker = ActionChunker(chunk_length=16)
    poses = _straight_line(20)
    with pytest.raises(IndexError, match=r'Valid anchors are 0\.\.3'):
        chunker.chunk(poses, np.zeros(20), 4)
    with pytest.raises(IndexError):
        chunker.chunk(poses, np.zeros(20), -1)


def test_malformed_inputs_are_refused():
    """Shape slips are caught here rather than producing a plausible chunk."""
    chunker = ActionChunker(chunk_length=2)
    with pytest.raises(ValueError, match=r'poses must be \(K, 7\)'):
        chunker.chunk(np.zeros((10, 6)), np.zeros(10), 0)
    with pytest.raises(ValueError, match='grippers must be'):
        chunker.chunk(_straight_line(10), np.zeros(9), 0)
    with pytest.raises(ValueError, match='chunk_length'):
        ActionChunker(chunk_length=0)
    with pytest.raises(ValueError, match="must be 'ee' or 'base'"):
        ActionChunker(frame='tool')


def test_growth_is_judged_per_channel_not_across_units():
    """
    Hold a shifted channel to its own baseline, not to another channel's.

    These are the real numbers from the 100 Hz run that exposed the bug:
    translation grew 124x and rotation 89x, both far beyond the 20x demanded.
    The old rule, min(shifted) > max(aligned) * 20, compared 7.277e-04 metres
    against 5.748e-05 * 20 -- a length against a dimensionless number -- and
    called that a failure.
    """
    aligned = (5.859e-06, 5.748e-05)   # metres, dimensionless
    shifted = (7.277e-04, 5.104e-03)
    assert channels_grew(shifted, aligned, 20.0)
    # The rule the bug hinged on, kept here so its wrongness stays visible.
    assert not min(shifted) > max(aligned) * 20.0


def test_a_channel_that_did_not_move_fails_even_if_another_moved_hugely():
    """Require all(), not any(): one channel hiding behind another is the risk."""
    aligned = (1e-06, 1e-06)
    assert not channels_grew((1.0, 2e-06), aligned, 20.0)
    assert not channels_grew((2e-06, 1.0), aligned, 20.0)
    assert channels_grew((1.0, 1.0), aligned, 20.0)


def test_growth_rejects_mismatched_and_empty_channel_lists():
    """Comparing different channel counts would silently pair the wrong ones."""
    with pytest.raises(ValueError):
        channels_grew((1.0, 2.0), (1e-6,), 20.0)
    assert not channels_grew((), (), 20.0)
