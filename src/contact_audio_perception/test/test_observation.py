"""Tests for the observation assembler: shapes, causality and every rejection."""
from contact_audio_perception.observation import (
    image_to_array,
    NANOSECONDS,
    ObservationAssembler,
    ObservationIncomplete,
    ObservationSpec,
    stamp_to_ns,
    TimedBuffer,
    transform_to_pose,
)
from contact_audio_perception.pose_math import pose_interpolate
import numpy as np
import pytest


def _spec(**overrides):
    """Build a small but realistic spec."""
    base = {
        'n_images': 2, 'image_shape': (4, 6, 3), 'audio_dim': 8,
        'audio_history_sec': 2.0, 'audio_rate_hz': 15.0,
    }
    base.update(overrides)
    return ObservationSpec(**base)


def _fill(assembler, reference_ns, audio_rate=15.0, image_rate=30.0, pose_rate=100.0,
          duration_sec=4.0, extra_sec=0.0):
    """
    Populate all three streams on a grid that lands exactly on the reference.

    Each stream carries its own index in its payload, so a test can check WHICH
    sample was selected rather than merely that one was. `extra_sec` continues
    the same grid PAST the reference, so causality can be tested against real
    future samples instead of against an empty buffer.
    """
    spec = assembler.spec
    start = reference_ns - int(duration_sec * NANOSECONDS)
    total = duration_sec + extra_sec
    for i in range(int(total * image_rate) + 1):
        stamp = start + int(i * NANOSECONDS / image_rate)
        assembler.add_image(stamp, np.full(spec.image_shape, i % 251, dtype=np.uint8))
    for i in range(int(total * audio_rate) + 1):
        stamp = start + int(i * NANOSECONDS / audio_rate)
        assembler.add_audio(stamp, np.full(spec.audio_dim, float(i), dtype=np.float32))
    for i in range(int(total * pose_rate) + 1):
        stamp = start + int(i * NANOSECONDS / pose_rate)
        t = stamp / NANOSECONDS
        assembler.add_pose(stamp, [0.1 * t, 0.2 * t, 0.3 * t, 0.0, 0.0, 0.0, 1.0])
    return start


# ---- TimedBuffer ----

def test_buffer_keeps_entries_ordered_even_when_they_arrive_late():
    """Out-of-order arrival must not put a stale sample where a fresh one goes."""
    buffer = TimedBuffer(10)
    for stamp in (30, 10, 50, 20, 40):
        buffer.add(stamp, stamp)
    assert buffer.stamps == [10, 20, 30, 40, 50]
    stamps, values = buffer.latest_at_or_before(45, 2)
    assert stamps == [30, 40] and values == [30, 40]


def test_buffer_drops_the_oldest_at_capacity():
    """A bounded buffer must discard from the front, not the back."""
    buffer = TimedBuffer(3)
    for stamp in range(6):
        buffer.add(stamp, stamp)
    assert buffer.stamps == [3, 4, 5]


def test_buffer_refuses_to_return_fewer_than_asked():
    """A short read would silently produce a ragged tensor, so it must raise."""
    buffer = TimedBuffer(10)
    buffer.add(10, 'a')
    with pytest.raises(ObservationIncomplete, match='need 2 entries'):
        buffer.latest_at_or_before(100, 2)


def test_buffer_refuses_to_extrapolate():
    """Interpolating past the last sample would invent robot state."""
    buffer = TimedBuffer(10)
    buffer.add(100, np.zeros(7))
    buffer.add(200, np.ones(7))
    with pytest.raises(ObservationIncomplete, match='extrapolation is refused'):
        buffer.sample_interpolated(250, pose_interpolate)
    with pytest.raises(ObservationIncomplete, match='extrapolation is refused'):
        buffer.sample_interpolated(50, pose_interpolate)


def test_buffer_reports_the_gap_to_the_nearest_sample():
    """The gap is the honest resolution of an interpolated channel."""
    buffer = TimedBuffer(10)
    buffer.add(0, np.array([0.0, 0, 0, 0, 0, 0, 1.0]))
    buffer.add(1000, np.array([10.0, 0, 0, 0, 0, 0, 1.0]))
    value, gap = buffer.sample_interpolated(200, pose_interpolate)
    assert np.isclose(value[0], 2.0)
    assert gap == 200
    _, gap_exact = buffer.sample_interpolated(1000, pose_interpolate)
    assert gap_exact == 0


# ---- shapes ----

def test_shapes_are_fixed_and_documented():
    """The declared shapes and the assembled ones must agree exactly."""
    spec = _spec()
    assembler = ObservationAssembler(spec)
    reference = 100 * NANOSECONDS
    _fill(assembler, reference)
    observation = assembler.assemble(reference)
    assert spec.shapes() == {
        'images': (2, 4, 6, 3), 'audio': (30, 8),
        'pose_current': (7,), 'pose_previous': (7,),
    }
    assert observation.images.shape == (2, 4, 6, 3)
    assert observation.images.dtype == np.uint8
    assert observation.audio.shape == (30, 8)
    assert observation.audio.dtype == np.float32
    assert observation.pose_current.shape == (7,)


def test_audio_count_follows_history_and_rate():
    """Deriving n_audio means changing the history changes the shape predictably."""
    assert _spec(audio_history_sec=2.0, audio_rate_hz=15.0).n_audio == 30
    assert _spec(audio_history_sec=1.0, audio_rate_hz=15.0).n_audio == 15
    assert _spec(audio_history_sec=3.0, audio_rate_hz=10.0).n_audio == 30


def test_default_buffers_cover_a_reference_that_lags_wall_clock():
    """
    The regression test for a bug the offline path could not have found.

    The reference is an audio window's header.stamp -- the capture time of that
    buffer's FIRST sample -- so it already trails wall-clock by the buffer
    length (1.0 s) before the encoder runs. Camera frames arriving "now" are
    therefore about a second NEWER than the instant being assembled.

    Buffers were originally sized from n_images (16 frames = 0.53 s at 30 Hz),
    which holds nothing old enough to use. Every offline test passed, because
    the offline script sizes its buffers from the length of the bag. The live
    node assembled exactly ZERO observations.

    So: feed the streams as they arrive in real time, assemble at a reference a
    second in the past, and require it to work with DEFAULT capacities.
    """
    spec = _spec()
    assembler = ObservationAssembler(spec)          # defaults, deliberately
    now = 100 * NANOSECONDS
    lag = NANOSECONDS                                # the 1.0 s audio buffer
    reference = now - lag

    start = now - 4 * NANOSECONDS
    for i in range(int(4 * 30) + 1):
        assembler.add_image(start + int(i * NANOSECONDS / 30),
                            np.full(spec.image_shape, i % 251, dtype=np.uint8))
    for i in range(int(4 * 15) + 1):
        assembler.add_audio(start + int(i * NANOSECONDS / 15),
                            np.full(spec.audio_dim, float(i), dtype=np.float32))
    for i in range(int(4 * 100) + 1):
        stamp = start + int(i * NANOSECONDS / 100)
        t = stamp / NANOSECONDS
        assembler.add_pose(stamp, [0.1 * t, 0.2 * t, 0.3 * t, 0.0, 0.0, 0.0, 1.0])

    observation = assembler.assemble(reference)
    assert observation.images.shape == (2,) + spec.image_shape
    assert bool(np.all(observation.image_stamps_ns <= reference))
    assert observation.audio_span_sec <= spec.audio_history_sec + 0.2


def test_buffer_capacity_is_derived_from_time_not_sample_count():
    """Sizing must follow the history each stream has to cover."""
    spec = _spec(image_rate_hz=30.0, audio_rate_hz=15.0, pose_rate_hz=100.0,
                 audio_history_sec=2.0, max_reference_lag_sec=2.0)
    assert spec.buffer_sec == 4.0
    assembler = ObservationAssembler(spec)
    # 4 s of history at 30 Hz is 120 frames; anything near n_images is wrong.
    assert assembler.images.capacity >= 120
    assert assembler.audio.capacity >= 60
    assert assembler.poses.capacity >= 400
    # Doubling the tolerated lag must double the buffers, not leave them alone.
    bigger = ObservationAssembler(_spec(max_reference_lag_sec=6.0))
    assert bigger.images.capacity > assembler.images.capacity * 1.8


# ---- causality and ordering ----

def test_nothing_after_the_reference_is_ever_used():
    """
    A future frame would not exist at run time, so using one poisons training.

    The payloads carry their index, so this checks the selected frames are the
    two immediately at-or-before the reference -- not merely that two arrived.
    """
    assembler = ObservationAssembler(
        _spec(), image_capacity=512, audio_capacity=512)
    reference = 100 * NANOSECONDS
    # A full second of real samples exists on the far side of the reference.
    _fill(assembler, reference, duration_sec=4.0, extra_sec=1.0)
    observation = assembler.assemble(reference)
    assert bool(np.all(observation.image_stamps_ns <= reference))
    assert bool(np.all(observation.audio_stamps_ns <= reference))
    # Exact identities, not just bounds: at 30 Hz the reference is frame 120,
    # so the two frames must be 119 and 120 -- never 121.
    assert list(observation.images[:, 0, 0, 0]) == [119, 120]
    # At 15 Hz the reference is item 60, and 30 items of history reach back to 31.
    assert observation.audio[0, 0] == 31.0
    assert observation.audio[-1, 0] == 60.0


def test_streams_are_ordered_oldest_first():
    """Index -1 is the most recent; reversing this silently reverses time."""
    assembler = ObservationAssembler(_spec())
    reference = 100 * NANOSECONDS
    _fill(assembler, reference)
    observation = assembler.assemble(reference)
    assert bool(np.all(np.diff(observation.image_stamps_ns) > 0))
    assert bool(np.all(np.diff(observation.audio_stamps_ns) > 0))
    assert observation.audio[-1, 0] > observation.audio[0, 0]
    assert observation.audio_stamps_ns[-1] == reference


def test_pose_pair_spans_exactly_pose_dt():
    """
    The two poses must be a fixed interval apart, not "the last two messages".

    With a straight-line motion the difference is dt times the velocity, so an
    assembler that returned the latest two samples (10 ms apart at 100 Hz)
    would be off by a factor of nearly seven here.
    """
    spec = _spec(pose_dt_sec=1.0 / 15.0)
    assembler = ObservationAssembler(spec)
    reference = 100 * NANOSECONDS
    _fill(assembler, reference)
    observation = assembler.assemble(reference)
    delta = observation.pose_current[:3] - observation.pose_previous[:3]
    expected = np.array([0.1, 0.2, 0.3]) * spec.pose_dt_sec
    assert np.allclose(delta, expected, atol=1e-9)


def test_pose_is_interpolated_between_samples_not_snapped_to_one():
    """The reference rarely coincides with a joint-state message."""
    assembler = ObservationAssembler(_spec())
    # 100 Hz poses on a 10 ms grid; ask for a pose halfway between two of them.
    grid = 100 * NANOSECONDS
    _fill(assembler, grid, extra_sec=0.02)
    reference = grid + 5 * 10 ** 6
    observation = assembler.assemble(reference)
    t = reference / NANOSECONDS
    assert np.allclose(observation.pose_current[:3], [0.1 * t, 0.2 * t, 0.3 * t],
                       atol=1e-9)
    assert 0 < observation.pose_gap_ns <= 5 * 10 ** 6


# ---- rejections ----

def test_a_frozen_camera_is_rejected_rather_than_repeated():
    """A dead camera must fail loudly, not feed one frame into every sample."""
    spec = _spec(max_image_age_sec=0.2)
    assembler = ObservationAssembler(spec)
    reference = 100 * NANOSECONDS
    _fill(assembler, reference - NANOSECONDS, duration_sec=4.0)
    for i in range(40):
        assembler.add_audio(reference - i * int(NANOSECONDS / 15),
                            np.zeros(8, dtype=np.float32))
        assembler.add_pose(reference - i * 10 ** 7, [0, 0, 0, 0, 0, 0, 1.0])
    with pytest.raises(ObservationIncomplete, match='camera frame is'):
        assembler.assemble(reference)


def test_dropped_audio_is_caught_by_the_span_check():
    """
    The fixed count hides a gap; the span check is what exposes it.

    Supplying only half the expected items still fills the (30, 8) tensor, so
    without this the history would silently cover 4 s instead of 2 s.
    """
    spec = _spec()
    assembler = ObservationAssembler(spec)
    reference = 100 * NANOSECONDS
    _fill(assembler, reference, audio_rate=7.5)
    with pytest.raises(ObservationIncomplete, match='audio history spans'):
        assembler.assemble(reference)


def test_a_sparse_pose_stream_is_rejected():
    """Interpolating across a long hole is extrapolation by another name."""
    spec = _spec(max_pose_gap_sec=0.05)
    assembler = ObservationAssembler(spec)
    reference = 100 * NANOSECONDS
    _fill(assembler, reference, pose_rate=2.0)
    with pytest.raises(ObservationIncomplete, match='nearest pose sample is'):
        assembler.assemble(reference)


def test_missing_streams_name_the_stream_that_failed():
    """Each rejection must name its stream: "could not assemble" is no diagnosis."""
    assembler = ObservationAssembler(_spec())
    with pytest.raises(ObservationIncomplete, match='need 2 entries'):
        assembler.assemble(100 * NANOSECONDS)


def test_wrong_shapes_are_refused_at_ingestion():
    """A mis-sized frame or vector is rejected where it enters, not later."""
    assembler = ObservationAssembler(_spec())
    with pytest.raises(ValueError, match='does not match the spec'):
        assembler.add_image(1, np.zeros((5, 6, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match='must be uint8'):
        assembler.add_image(1, np.zeros((4, 6, 3), dtype=np.float32))
    with pytest.raises(ValueError, match='spec says'):
        assembler.add_audio(1, np.zeros(9, dtype=np.float32))
    with pytest.raises(ValueError, match='pose must be 7 values'):
        assembler.add_pose(1, [0, 0, 0, 1])


def test_spec_rejects_impossible_configuration():
    """Nonsense configuration fails at construction, not at the first sample."""
    with pytest.raises(ValueError, match='n_images'):
        _spec(n_images=0)
    with pytest.raises(ValueError, match='audio_rate_hz'):
        _spec(audio_rate_hz=0.0)
    with pytest.raises(ValueError, match='pose_dt_sec'):
        _spec(pose_dt_sec=-1.0)
    with pytest.raises(ValueError, match='image_shape'):
        _spec(image_shape=(4, 6))


# ---- message adapters ----

class _FakeImage:
    """Minimal stand-in for sensor_msgs/Image."""

    def __init__(self, height, width, encoding, data, step=None):
        """Store the fields image_to_array reads."""
        self.height = height
        self.width = width
        self.encoding = encoding
        self.data = data
        self.step = step if step is not None else len(data) // height


def test_image_adapter_honours_a_padded_row_stride():
    """
    A `step` wider than width*channels is legal and shears the image if ignored.

    Reshaping straight to (h, w, c) is the bug this guards: it would slide each
    row left by the padding and still produce an array of the right shape.
    """
    height, width, pad = 3, 4, 5
    step = width * 3 + pad
    rows = []
    for row in range(height):
        rows.append(bytes([row * 10 + i for i in range(width * 3)]) + bytes(pad))
    msg = _FakeImage(height, width, 'rgb8', b''.join(rows), step=step)
    array = image_to_array(msg)
    assert array.shape == (3, 4, 3)
    assert array[2, 0, 0] == 20
    assert array[1, 3, 2] == 10 + 11


def test_image_adapter_refuses_an_unknown_encoding():
    """Guessing an encoding turns BGR into RGB invisibly."""
    msg = _FakeImage(1, 1, '16UC1', bytes(2))
    with pytest.raises(ValueError, match='unsupported image encoding'):
        image_to_array(msg)


def test_image_adapter_refuses_a_truncated_payload():
    """A short buffer means a dropped fragment, not an image."""
    msg = _FakeImage(2, 2, 'rgb8', bytes(6), step=6)
    with pytest.raises(ValueError, match='header implies'):
        image_to_array(msg)


class _FakeHeader:
    """Minimal stand-in for std_msgs/Header."""

    class _Stamp:
        def __init__(self, sec, nanosec):
            """Store one builtin_interfaces/Time."""
            self.sec, self.nanosec = sec, nanosec

    def __init__(self, sec, nanosec):
        """Wrap a stamp the way a real header does."""
        self.stamp = self._Stamp(sec, nanosec)


def test_stamp_to_ns_takes_a_header_not_a_time():
    """
    The convention is pinned: callers hold headers, so the function takes one.

    Passing a Header to a function expecting a Time raises AttributeError, which
    is loud -- but the reverse convention would make every caller wrong at once.
    """
    assert stamp_to_ns(_FakeHeader(7, 250_000_000)) == 7 * NANOSECONDS + 250_000_000


class _FakeTransform:
    """Minimal stand-in for geometry_msgs/Transform."""

    class _Vec:
        def __init__(self, x, y, z, w=None):
            """Store one vector or quaternion."""
            self.x, self.y, self.z = x, y, z
            if w is not None:
                self.w = w

    def __init__(self):
        """Build a transform with distinguishable components."""
        self.translation = self._Vec(1.0, 2.0, 3.0)
        self.rotation = self._Vec(0.1, 0.2, 0.3, 0.9)


def test_transform_adapter_keeps_ros_xyzw_order():
    """
    The scalar part stays LAST, matching geometry_msgs/Quaternion.

    Silently reordering to WXYZ is invisible in every shape check and wrong in
    every rotation.
    """
    pose = transform_to_pose(_FakeTransform())
    assert np.allclose(pose, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9])
