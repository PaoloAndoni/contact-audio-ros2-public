"""
Assemble what a policy consumes, at one reference instant, from three streams.

WHY A LIBRARY AND NOT A NODE: the same assembly has to happen live (a running
policy) and offline (building a dataset from a bag). If those were two
implementations, the policy would be trained on observations it never sees at
run time, and nothing in the graph would say so. So the rules live here, ROS
messages are converted at the edge, and both callers get the same object.

THE REFERENCE INSTANT is an audio window's header.stamp -- the capture time of
its FIRST sample. Every other stream is resolved relative to that. Audio is the
reference and not the camera because the audio stamp is the one the whole
pipeline has been built to keep honest.

SELECTION POLICY, per stream. All three are STRICTLY CAUSAL: nothing with a
stamp after the reference may be used, because at run time it would not exist
yet. A dataset built from future data trains a policy that cannot be deployed.

  CAMERA   nearest-preceding, no interpolation. The `n_images` frames with the
           largest stamps <= reference, ordered OLDEST FIRST so index -1 is the
           most recent. Images are not interpolated: averaging two frames makes
           a picture of a scene that never existed. Rejected if the newest
           frame is older than `max_image_age_sec` -- otherwise a dead camera
           silently feeds one frozen frame into every observation.

  AUDIO    fixed-length history, no interpolation. The `n_audio` items with the
           largest stamps <= reference, oldest first, where
           `n_audio = round(history_sec * nominal_rate)`. The count is fixed by
           configuration rather than by what happened to arrive, so the tensor
           shape is constant; a dropped item is then caught by the span check
           (the oldest must be no further back than history_sec plus
           `audio_span_tolerance`) instead of quietly compressing the history.
           Default history is 2.0 s: ManiWAV reports 2-3 s beating 1 s on
           contact-rich tasks.

  POSE     INTERPOLATED, because it is the one stream that can be. Joint states
           arrive at ~100 Hz asynchronously, so the pose exactly at the
           reference is almost never a sample. Translation is linear, rotation
           is slerp (component-wise lerp would move at a non-constant angular
           rate). Extrapolation is refused: the reference must lie between two
           samples. `pose_previous` is the same interpolation at
           `reference - pose_dt_sec`, so the pair is a true fixed-interval
           difference and not "whatever the last message happened to be".

OUTPUT SHAPES are fixed and asserted before returning:
    images         (n_images, height, width, channels)  uint8
    audio          (n_audio, audio_dim)                 float32
    pose_current   (7,)                                 float64  [xyz + xyzw]
    pose_previous  (7,)                                 float64  [xyz + xyzw]
"""
import bisect
from dataclasses import dataclass
import math
import re

from contact_audio_perception.pose_math import pose_interpolate
import numpy as np

NANOSECONDS = 1_000_000_000


def failure_category(exception):
    """
    Return a rejection reason with its timestamps masked, for counting.

    ObservationIncomplete names the exact stamp that failed, which is what you
    want when reading one message and exactly what you do not want when
    tallying a thousand: every drop becomes its own category and the count that
    matters disappears.
    """
    text = str(exception).split(',')[0].split(';')[0]
    return re.sub(r'\d{6,}', '<stamp>', text)


DEFAULT_N_IMAGES = 2
DEFAULT_AUDIO_HISTORY_SEC = 2.0
DEFAULT_AUDIO_RATE_HZ = 15.0
# One audio step. The pose pair then spans exactly the interval between two
# observations, which is what makes it comparable with a derived action.
DEFAULT_POSE_DT_SEC = 1.0 / DEFAULT_AUDIO_RATE_HZ


class ObservationIncomplete(Exception):
    """Raised when the streams cannot cover the requested reference instant."""


@dataclass(frozen=True)
class Observation:
    """One assembled observation, with the provenance needed to audit it."""

    reference_stamp_ns: int
    images: np.ndarray
    image_stamps_ns: np.ndarray
    audio: np.ndarray
    audio_stamps_ns: np.ndarray
    pose_current: np.ndarray
    pose_previous: np.ndarray
    pose_dt_sec: float
    # Worst distance from an interpolation target to its nearest real sample.
    # Reported, not hidden: it is the honest resolution of the pose channel.
    pose_gap_ns: int

    @property
    def image_age_sec(self):
        """Return how stale the newest camera frame is, in seconds."""
        return (self.reference_stamp_ns - int(self.image_stamps_ns[-1])) / NANOSECONDS

    @property
    def audio_span_sec(self):
        """Return how far back the audio history actually reaches, in seconds."""
        return (self.reference_stamp_ns - int(self.audio_stamps_ns[0])) / NANOSECONDS


class TimedBuffer:
    """
    A bounded, time-ordered buffer of (stamp_ns, value).

    Insertion is sorted rather than append-only. ROS delivers messages out of
    order often enough -- two publishers, a bag replayed with `--rate`, a node
    that batches -- that assuming monotonic arrival puts a stale sample where a
    fresh one belongs, and the bisect that selects frames then returns the
    wrong ones without ever failing.
    """

    def __init__(self, capacity):
        """Create a buffer holding at most `capacity` most-recent entries."""
        if capacity < 1:
            raise ValueError(f'capacity must be >= 1, got {capacity}')
        self.capacity = int(capacity)
        self._stamps = []
        self._values = []

    def __len__(self):
        """Return how many entries are held."""
        return len(self._stamps)

    @property
    def stamps(self):
        """Return the held timestamps, ascending."""
        return list(self._stamps)

    def add(self, stamp_ns, value):
        """Insert one entry, dropping the oldest once capacity is exceeded."""
        stamp_ns = int(stamp_ns)
        index = bisect.bisect_right(self._stamps, stamp_ns)
        self._stamps.insert(index, stamp_ns)
        self._values.insert(index, value)
        while len(self._stamps) > self.capacity:
            self._stamps.pop(0)
            self._values.pop(0)

    def latest_at_or_before(self, stamp_ns, count):
        """
        Return the `count` newest entries with stamp <= stamp_ns, oldest first.

        Raises ObservationIncomplete rather than returning a short list: a
        caller that silently accepted fewer would produce a ragged tensor.
        """
        end = bisect.bisect_right(self._stamps, int(stamp_ns))
        if end < count:
            raise ObservationIncomplete(
                f'need {count} entries at or before {stamp_ns}, have {end} '
                f'(buffer holds {len(self._stamps)})')
        start = end - count
        return self._stamps[start:end], self._values[start:end]

    def sample_interpolated(self, stamp_ns, interpolate):
        """
        Return (value, gap_ns) at an arbitrary instant, refusing to extrapolate.

        `gap_ns` is the distance to the nearer bracketing sample, so a caller
        can tell a dense interpolation from one that bridged a hole.
        """
        stamp_ns = int(stamp_ns)
        if len(self._stamps) < 2:
            raise ObservationIncomplete(
                f'need at least two samples to interpolate, have {len(self._stamps)}')
        if stamp_ns < self._stamps[0] or stamp_ns > self._stamps[-1]:
            raise ObservationIncomplete(
                f'{stamp_ns} lies outside the buffered span '
                f'[{self._stamps[0]}, {self._stamps[-1]}]; extrapolation is refused')
        index = bisect.bisect_left(self._stamps, stamp_ns)
        if self._stamps[index] == stamp_ns:
            return self._values[index], 0
        before, after = self._stamps[index - 1], self._stamps[index]
        span = after - before
        fraction = 0.0 if span == 0 else (stamp_ns - before) / span
        value = interpolate(self._values[index - 1], self._values[index], fraction)
        return value, int(min(stamp_ns - before, after - stamp_ns))


@dataclass
class ObservationSpec:
    """The fixed shape and timing contract of an assembled observation."""

    n_images: int = DEFAULT_N_IMAGES
    image_shape: tuple = (120, 160, 3)
    audio_dim: int = 128
    audio_history_sec: float = DEFAULT_AUDIO_HISTORY_SEC
    audio_rate_hz: float = DEFAULT_AUDIO_RATE_HZ
    # A dropped audio item stretches the span. 1.5 steps of slack tolerates
    # jitter while still catching a real gap.
    audio_span_tolerance_sec: float = 1.5 / DEFAULT_AUDIO_RATE_HZ
    pose_dt_sec: float = DEFAULT_POSE_DT_SEC
    max_image_age_sec: float = 0.5
    max_pose_gap_sec: float = 0.05
    # Nominal arrival rates of the two streams that are NOT the clock. Used only
    # to size buffers, never to select samples -- selection always goes by the
    # stamps that actually arrived.
    image_rate_hz: float = 30.0
    pose_rate_hz: float = 100.0
    # HOW FAR BEHIND WALL-CLOCK THE REFERENCE CAN BE, and the reason buffers
    # cannot be sized by sample count alone. The reference is an audio window's
    # header.stamp, i.e. the capture time of that buffer's FIRST sample, so it
    # already trails real time by the buffer length (1.0 s) before the encoder
    # has done anything. Camera frames arriving now are therefore a second
    # NEWER than the instant being assembled, and a buffer holding only the
    # newest few frames holds nothing old enough to use. Sizing from n_images
    # looked reasonable and made the live node assemble exactly zero
    # observations while every offline test passed.
    max_reference_lag_sec: float = 2.0

    def __post_init__(self):
        """Validate the spec and derive the fixed audio history length."""
        if self.n_images < 1:
            raise ValueError(f'n_images must be >= 1, got {self.n_images}')
        if self.audio_rate_hz <= 0.0:
            raise ValueError(f'audio_rate_hz must be > 0, got {self.audio_rate_hz}')
        if self.audio_history_sec <= 0.0:
            raise ValueError(
                f'audio_history_sec must be > 0, got {self.audio_history_sec}')
        if self.pose_dt_sec <= 0.0:
            raise ValueError(f'pose_dt_sec must be > 0, got {self.pose_dt_sec}')
        self.image_shape = tuple(int(v) for v in self.image_shape)
        if len(self.image_shape) != 3:
            raise ValueError(
                f'image_shape must be (height, width, channels), got {self.image_shape}')

    @property
    def buffer_sec(self):
        """Return how much history every stream must hold to cover one assembly."""
        return self.audio_history_sec + self.max_reference_lag_sec

    def default_capacity(self, rate_hz, minimum):
        """Return a buffer length covering buffer_sec at `rate_hz`, with headroom."""
        return max(int(minimum), int(math.ceil(self.buffer_sec * rate_hz * 1.5)))

    @property
    def n_audio(self):
        """Return the fixed number of audio items in one observation."""
        return max(1, int(round(self.audio_history_sec * self.audio_rate_hz)))

    def shapes(self):
        """Return the exact output shapes, for a consumer to assert against."""
        return {
            'images': (self.n_images,) + self.image_shape,
            'audio': (self.n_audio, int(self.audio_dim)),
            'pose_current': (7,),
            'pose_previous': (7,),
        }


class ObservationAssembler:
    """Collect three asynchronous streams and assemble observations from them."""

    def __init__(self, spec=None, image_capacity=None, audio_capacity=None,
                 pose_capacity=None):
        """
        Size every buffer from the spec, with generous headroom.

        Sized by TIME, not by sample count: each buffer must reach back
        max_reference_lag_sec + audio_history_sec, because the reference instant
        is already about a second behind wall-clock when it arrives. See
        ObservationSpec.max_reference_lag_sec for what happens otherwise.
        """
        self.spec = spec or ObservationSpec()
        self.images = TimedBuffer(image_capacity or self.spec.default_capacity(
            self.spec.image_rate_hz, self.spec.n_images * 4))
        self.audio = TimedBuffer(audio_capacity or self.spec.default_capacity(
            self.spec.audio_rate_hz, self.spec.n_audio * 2))
        self.poses = TimedBuffer(pose_capacity or self.spec.default_capacity(
            self.spec.pose_rate_hz, 256))

    # ---- ingestion ----

    def add_image(self, stamp_ns, image):
        """Buffer one camera frame, checking its shape against the spec."""
        image = np.asarray(image)
        if image.shape != self.spec.image_shape:
            raise ValueError(
                f'image shape {image.shape} does not match the spec '
                f'{self.spec.image_shape}')
        if image.dtype != np.uint8:
            raise ValueError(f'images must be uint8, got {image.dtype}')
        self.images.add(stamp_ns, image)

    def add_audio(self, stamp_ns, vector):
        """Buffer one audio feature vector (an embedding, or a flat spectrogram)."""
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != int(self.spec.audio_dim):
            raise ValueError(
                f'audio vector has {vector.shape[0]} values, spec says '
                f'{self.spec.audio_dim}')
        self.audio.add(stamp_ns, vector)

    def add_pose(self, stamp_ns, pose):
        """Buffer one end-effector pose as [x, y, z, qx, qy, qz, qw]."""
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape[0] != 7:
            raise ValueError(
                f'pose must be 7 values [x,y,z,qx,qy,qz,qw], got {pose.shape[0]}')
        self.poses.add(stamp_ns, pose)

    # ---- assembly ----

    def assemble(self, reference_stamp_ns):
        """
        Assemble the observation at one reference instant.

        Raises ObservationIncomplete with a specific reason whenever any stream
        cannot cover it. Every rejection names the stream and the number that
        failed, because "could not assemble" is not a diagnosis.
        """
        reference_stamp_ns = int(reference_stamp_ns)
        spec = self.spec

        image_stamps, image_values = self.images.latest_at_or_before(
            reference_stamp_ns, spec.n_images)
        image_age = (reference_stamp_ns - image_stamps[-1]) / NANOSECONDS
        if image_age > spec.max_image_age_sec:
            raise ObservationIncomplete(
                f'newest camera frame is {image_age * 1e3:.1f} ms old, limit is '
                f'{spec.max_image_age_sec * 1e3:.1f} ms')

        audio_stamps, audio_values = self.audio.latest_at_or_before(
            reference_stamp_ns, spec.n_audio)
        span = (reference_stamp_ns - audio_stamps[0]) / NANOSECONDS
        limit = spec.audio_history_sec + spec.audio_span_tolerance_sec
        if span > limit:
            raise ObservationIncomplete(
                f'audio history spans {span:.3f} s, more than '
                f'{limit:.3f} s: items were dropped, so the {spec.n_audio} newest '
                f'do not cover {spec.audio_history_sec:.3f} s contiguously')

        previous_ns = reference_stamp_ns - int(round(spec.pose_dt_sec * NANOSECONDS))
        pose_current, gap_current = self.poses.sample_interpolated(
            reference_stamp_ns, pose_interpolate)
        pose_previous, gap_previous = self.poses.sample_interpolated(
            previous_ns, pose_interpolate)
        pose_gap = max(gap_current, gap_previous)
        if pose_gap > spec.max_pose_gap_sec * NANOSECONDS:
            raise ObservationIncomplete(
                f'nearest pose sample is {pose_gap / 1e6:.1f} ms away, limit is '
                f'{spec.max_pose_gap_sec * 1e3:.1f} ms')

        observation = Observation(
            reference_stamp_ns=reference_stamp_ns,
            images=np.stack(image_values).astype(np.uint8, copy=False),
            image_stamps_ns=np.array(image_stamps, dtype=np.int64),
            audio=np.stack(audio_values).astype(np.float32, copy=False),
            audio_stamps_ns=np.array(audio_stamps, dtype=np.int64),
            pose_current=np.asarray(pose_current, dtype=np.float64),
            pose_previous=np.asarray(pose_previous, dtype=np.float64),
            pose_dt_sec=float(spec.pose_dt_sec),
            pose_gap_ns=int(pose_gap),
        )
        expected = spec.shapes()
        actual = {
            'images': observation.images.shape,
            'audio': observation.audio.shape,
            'pose_current': observation.pose_current.shape,
            'pose_previous': observation.pose_previous.shape,
        }
        if actual != expected:
            raise ObservationIncomplete(
                f'assembled shapes {actual} do not match the spec {expected}')
        return observation


# ---- ROS message adapters, kept at the edge so the logic above stays testable ----

def stamp_to_ns(header):
    """
    Return a std_msgs/Header's stamp as integer nanoseconds.

    Takes the HEADER, not the stamp, matching align_audio_robot.py's stamp_ns()
    -- every caller has a header in hand, and the two conventions living side by
    side is how a header gets passed to a function expecting a Time and silently
    reads the wrong field.
    """
    return int(header.stamp.sec) * NANOSECONDS + int(header.stamp.nanosec)


def image_to_array(msg):
    """
    Convert sensor_msgs/Image to an (H, W, C) uint8 array without cv_bridge.

    Only the 8-bit encodings a policy actually uses are accepted. Guessing at an
    unknown encoding is how a BGR image quietly becomes an RGB one, which is
    invisible in every test and wrong in every inference.
    """
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}
    if msg.encoding not in channels:
        raise ValueError(
            f'unsupported image encoding {msg.encoding!r}; supported: '
            f'{sorted(channels)}')
    count = channels[msg.encoding]
    array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    expected = int(msg.height) * int(msg.step)
    if array.size != expected:
        raise ValueError(
            f'image data is {array.size} bytes, header implies {expected} '
            f'({msg.height} rows x {msg.step} bytes)')
    # Honour `step`: a padded row stride is legal and silently shears the image
    # if the data is reshaped to width*channels instead.
    array = array.reshape(int(msg.height), int(msg.step))
    return array[:, :int(msg.width) * count].reshape(
        int(msg.height), int(msg.width), count)


def transform_to_pose(transform):
    """Return [x, y, z, qx, qy, qz, qw] from a geometry_msgs/Transform."""
    t = transform.translation
    r = transform.rotation
    return np.array([t.x, t.y, t.z, r.x, r.y, r.z, r.w], dtype=np.float64)
