"""
Turn a recorded bag into (observation, action) pairs for policy learning.

    bag -> per-step observations (images + audio history + pose pair)
        -> per-step action chunks (16 x 10, relative end-effector deltas)
        -> one .npz

STEP CLOCK: the audio topic. Step k is anchored at audio message k's
header.stamp -- the capture time of that buffer's FIRST sample, which the whole
pipeline exists to keep honest. Everything else is resolved against it by
contact_audio_perception.observation, so what this script builds offline is
what observation_node assembles live, by the same code.

END-EFFECTOR POSE COMES FROM /tf, not from forward kinematics. Three reasons,
in order of weight:

  1. It is the only representation the three possible collection methods share.
     A hand-guided arm and a teleoperated arm publish joint states; a handheld
     gripper (UMI-style, which is what ManiWAV used) has no joints at all and
     publishes only a pose. FK would work for two of the three and the choice
     of collection method is not yet made.
  2. FK needs a URDF and the exact kinematic parameters of the real arm,
     including any tool transform. We do not have them, and an FK chain that is
     subtly wrong produces poses that are smooth, plausible and displaced --
     the same failure mode this script's verification is built to catch.
  3. /tf is what the robot itself asserts its pose to be, rather than our
     reconstruction of it.

If a robot publishes only /joint_states, the standard answer is to run
robot_state_publisher with its URDF, which puts base->tool on /tf. This script
then works unchanged; the FK happens where the URDF lives.

ACTION REPRESENTATION: ManiWAV's 10 numbers -- 3 translation + 6-D rotation + 1
gripper -- as a chunk of 16 future steps. Matched rather than invented so our
numbers stay comparable with the published contact-audio result. The layout and
the step-index contract are defined in contact_audio_perception.action; read
that docstring before changing anything here.

COLLECTION METHOD IS NOT ASSUMED. Deltas default to the end-effector's own
frame, which is invariant to where the base sits, so hand-guiding, teleoperation
and a handheld device all derive identically. If a teleop command topic is
present it is recorded as well and preferred, but it is never required: pass
--command-topic to use one, and the output records which source each action
came from.

Usage:
    ros2 bag record -s mcap -o run1 /audio/embedding /audio/spectrogram \\
        /camera/image_raw /tf /tf_static /joint_states
    python3 scripts/derive_actions.py run1 --out run1_actions.npz
"""
import argparse
import os
import sys

import numpy as np
import rclpy.time
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'src',
    'contact_audio_perception'))
from contact_audio_perception.action import ActionChunker  # noqa: E402
from contact_audio_perception.observation import (  # noqa: E402
    failure_category,
    image_to_array,
    NANOSECONDS,
    ObservationAssembler,
    ObservationIncomplete,
    ObservationSpec,
    stamp_to_ns,
    transform_to_pose,
)
from contact_audio_perception.pose_math import pose_interpolate  # noqa: E402

DEFAULT_AUDIO_TOPIC = '/audio/embedding'
DEFAULT_IMAGE_TOPIC = '/camera/image_raw'


def read_bag(path, topics):
    """Yield (topic, deserialised message) for the requested topics."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id=''),
        rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                    output_serialization_format='cdr'))
    available = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = [t for t in topics if t in available]
    missing = [t for t in topics if t not in available]
    if missing:
        print(f'  note: bag has no {missing} (it has {sorted(available)})')
    if not wanted:
        raise SystemExit(f'{path}: none of {topics} are in the bag')
    reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        yield topic, deserialize_message(raw, get_message(available[topic]))


def poses_from_tf(bag, base_frame, tool_frame):
    """
    Replay /tf into a tf2 buffer and return (stamps_ns, poses) for base->tool.

    Every distinct /tf stamp is tried as a lookup. For a single broadcast
    transform that is exact; for a real multi-link chain, robot_state_publisher
    stamps all links together, so the same stamps resolve the whole chain.
    Stamps that cannot be resolved are skipped and counted, never guessed at.
    """
    # A long cache: the default 10 s would silently drop the start of any bag
    # longer than that, and the loss would look like a sparse pose stream.
    buffer = Buffer(cache_time=Duration(seconds=24 * 3600))
    stamps = set()
    for topic, msg in read_bag(bag, ['/tf', '/tf_static']):
        static = topic == '/tf_static'
        for transform in msg.transforms:
            if static:
                buffer.set_transform_static(transform, 'derive_actions')
            else:
                buffer.set_transform(transform, 'derive_actions')
                stamps.add(stamp_to_ns(transform.header))

    resolved_stamps, resolved_poses, failed = [], [], 0
    for stamp in sorted(stamps):
        try:
            transform = buffer.lookup_transform(
                base_frame, tool_frame, rclpy.time.Time(nanoseconds=stamp))
        except Exception:
            failed += 1
            continue
        resolved_stamps.append(stamp)
        resolved_poses.append(transform_to_pose(transform.transform))
    if failed:
        print(f'  {failed} /tf stamps could not resolve {base_frame} -> {tool_frame}')
    return resolved_stamps, resolved_poses


def gripper_series(bag, joint_name):
    """Return (stamps_ns, values) for a gripper joint in /joint_states, or None."""
    if not joint_name:
        return None
    stamps, values = [], []
    for _, msg in read_bag(bag, ['/joint_states']):
        if joint_name in msg.name:
            stamps.append(stamp_to_ns(msg.header))
            values.append(float(msg.position[list(msg.name).index(joint_name)]))
    if not stamps:
        print(f'  no joint {joint_name!r} in /joint_states; gripper channel stays 0')
        return None
    order = np.argsort(np.array(stamps, dtype=np.int64))
    return np.array(stamps, dtype=np.int64)[order], np.array(values)[order]


def commanded_poses(bag, topic):
    """
    Return (stamps_ns, poses) from a teleop command topic, or None.

    Accepts geometry_msgs/PoseStamped and geometry_msgs/TransformStamped, the
    two shapes a commanded end-effector target actually arrives in. Anything
    else is reported and skipped: a command topic is a bonus, never a
    requirement, so an unrecognised one must not fail the derivation.
    """
    if not topic:
        return None
    stamps, poses = [], []
    for _, msg in read_bag(bag, [topic]):
        if hasattr(msg, 'pose'):
            p, r = msg.pose.position, msg.pose.orientation
        elif hasattr(msg, 'transform'):
            p, r = msg.transform.translation, msg.transform.rotation
        else:
            print(f'  {topic}: unsupported message type {type(msg).__name__}; '
                  f'commanded actions will not be derived')
            return None
        stamps.append(stamp_to_ns(msg.header))
        poses.append([p.x, p.y, p.z, r.x, r.y, r.z, r.w])
    if not stamps:
        return None
    order = np.argsort(np.array(stamps, dtype=np.int64))
    return (np.array(stamps, dtype=np.int64)[order],
            np.array(poses, dtype=np.float64)[order])


def sample_poses_at(pose_buffer, step_stamps):
    """Interpolate the pose at every step time; return (poses, ok_mask, gaps)."""
    poses = np.zeros((len(step_stamps), 7))
    ok = np.zeros(len(step_stamps), dtype=bool)
    gaps = np.full(len(step_stamps), -1, dtype=np.int64)
    for i, stamp in enumerate(step_stamps):
        try:
            pose, gap = pose_buffer.sample_interpolated(stamp, pose_interpolate)
        except ObservationIncomplete:
            continue
        poses[i] = pose
        gaps[i] = gap
        ok[i] = True
    return poses, ok, gaps


def main():
    """Derive the dataset and write it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag')
    parser.add_argument('--out', default='actions.npz')
    parser.add_argument('--audio-topic', default=DEFAULT_AUDIO_TOPIC)
    parser.add_argument('--image-topic', default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument('--base-frame', default='base_link')
    parser.add_argument('--tool-frame', default='tool0')
    parser.add_argument('--chunk-length', type=int, default=16)
    parser.add_argument('--frame', choices=('ee', 'base'), default='ee',
                        help="frame the deltas are expressed in (default: ee)")
    parser.add_argument('--n-images', type=int, default=2)
    parser.add_argument('--audio-history-sec', type=float, default=2.0)
    parser.add_argument('--audio-rate-hz', type=float, default=15.0)
    parser.add_argument('--pose-dt-sec', type=float, default=0.0,
                        help='0 = one audio step')
    parser.add_argument('--max-pose-gap-sec', type=float, default=0.05)
    parser.add_argument('--max-image-age-sec', type=float, default=0.5)
    parser.add_argument('--gripper-joint', default='',
                        help='joint in /joint_states to use as the gripper channel')
    parser.add_argument('--command-topic', default='',
                        help='teleop command topic; recorded and preferred if present')
    parser.add_argument('--no-images', action='store_true',
                        help='omit the image arrays from the output (much smaller)')
    args = parser.parse_args()

    print(f'bag: {args.bag}')
    pose_stamps, pose_values = poses_from_tf(args.bag, args.base_frame, args.tool_frame)
    if len(pose_stamps) < 2:
        raise SystemExit(
            f'need at least two {args.base_frame} -> {args.tool_frame} transforms, '
            f'got {len(pose_stamps)}')
    span = (pose_stamps[-1] - pose_stamps[0]) / NANOSECONDS
    print(f'poses              : {len(pose_stamps)} over {span:.2f} s '
          f'({(len(pose_stamps) - 1) / max(span, 1e-9):.1f} Hz)')

    audio, images = [], []
    for topic, msg in read_bag(args.bag, [args.audio_topic, args.image_topic]):
        (audio if topic == args.audio_topic else images).append(msg)
    if not audio:
        raise SystemExit(f'no messages on {args.audio_topic}')
    if not images:
        raise SystemExit(f'no messages on {args.image_topic}')
    audio.sort(key=lambda m: stamp_to_ns(m.header))
    print(f'audio steps        : {len(audio)} on {args.audio_topic}')
    print(f'camera frames      : {len(images)} on {args.image_topic}')

    first_vector = np.asarray(getattr(audio[0], 'embedding', None)
                              if hasattr(audio[0], 'embedding') else audio[0].data,
                              dtype=np.float32).reshape(-1)
    first_image = image_to_array(images[0])
    rate = args.audio_rate_hz
    spec = ObservationSpec(
        n_images=args.n_images, image_shape=first_image.shape,
        audio_dim=first_vector.size, audio_history_sec=args.audio_history_sec,
        audio_rate_hz=rate, audio_span_tolerance_sec=1.5 / rate,
        pose_dt_sec=args.pose_dt_sec if args.pose_dt_sec > 0 else 1.0 / rate,
        max_image_age_sec=args.max_image_age_sec,
        max_pose_gap_sec=args.max_pose_gap_sec)
    print(f'observation shapes : {spec.shapes()}')

    assembler = ObservationAssembler(
        spec, image_capacity=max(64, len(images) + 1),
        audio_capacity=max(64, len(audio) + 1),
        pose_capacity=max(64, len(pose_stamps) + 1))
    for stamp, pose in zip(pose_stamps, pose_values):
        assembler.add_pose(stamp, pose)
    for msg in images:
        assembler.add_image(stamp_to_ns(msg.header), image_to_array(msg))
    for msg in audio:
        vector = (np.asarray(msg.embedding, dtype=np.float32)
                  if hasattr(msg, 'embedding')
                  else np.asarray(msg.data, dtype=np.float32))
        assembler.add_audio(stamp_to_ns(msg.header), vector)

    # STEP CLOCK. Kept for EVERY audio message, including steps that fail to
    # assemble, because verification has to be able to recompute ground truth at
    # the same step indices the derivation used. Dropping them here is how an
    # off-by-one becomes invisible.
    step_stamps = np.array([stamp_to_ns(m.header) for m in audio], dtype=np.int64)
    step_poses, pose_ok, pose_gaps = sample_poses_at(assembler.poses, step_stamps)
    print(f'steps with a pose  : {int(pose_ok.sum())}/{len(step_stamps)}')

    gripper = np.zeros(len(step_stamps))
    gripper_source = 'none'
    series = gripper_series(args.bag, args.gripper_joint)
    if series is not None:
        gripper = np.interp(step_stamps, series[0], series[1])
        gripper_source = f'joint:{args.gripper_joint}'
    if gripper_source == 'none':
        print('gripper            : no source; the channel is ZERO, not measured')

    chunker = ActionChunker(chunk_length=args.chunk_length, frame=args.frame)
    n_candidates = chunker.valid_samples(len(step_stamps))
    if n_candidates == 0:
        raise SystemExit(
            f'{len(step_stamps)} steps is too few for a chunk of '
            f'{args.chunk_length}; record a longer bag')

    commanded = commanded_poses(args.bag, args.command_topic)
    commanded_actions = None
    action_source = 'tf_pose'
    if commanded is not None:
        command_buffer = ObservationAssembler(
            spec, pose_capacity=max(64, len(commanded[0]) + 1)).poses
        for stamp, pose in zip(commanded[0], commanded[1]):
            command_buffer.add(int(stamp), np.asarray(pose, dtype=np.float64))
        command_poses, command_ok, _ = sample_poses_at(command_buffer, step_stamps)
        if command_ok.all():
            commanded_actions = command_poses
            action_source = 'teleop_command'
            print(f'commanded actions  : from {args.command_topic} (preferred)')
        else:
            print(f'  {args.command_topic} does not cover every step '
                  f'({int(command_ok.sum())}/{len(step_stamps)}); using /tf poses')

    # A sample is emitted only when its observation assembles AND every pose its
    # chunk needs exists. Half a chunk is not a short chunk, it is a wrong one.
    rows, reasons = [], {}
    for k in range(n_candidates):
        if not pose_ok[k:k + args.chunk_length + 1].all():
            reasons['missing pose in the chunk'] = \
                reasons.get('missing pose in the chunk', 0) + 1
            continue
        try:
            observation = assembler.assemble(int(step_stamps[k]))
        except ObservationIncomplete as exc:
            key = failure_category(exc)
            reasons[key] = reasons.get(key, 0) + 1
            continue
        rows.append((k, observation))

    print(f'candidate steps    : {n_candidates} (of {len(step_stamps)}; the last '
          f'{args.chunk_length} cannot start a chunk)')
    print(f'samples emitted    : {len(rows)}')
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f'  dropped {count:4d}: {reason}')
    if not rows:
        raise SystemExit('nothing could be derived')

    indices = np.array([k for k, _ in rows], dtype=np.int64)
    observations = [o for _, o in rows]
    actions = np.stack([chunker.chunk(step_poses, gripper, int(k)) for k in indices])
    payload = {
        'sample_step_index': indices,
        'step_stamp_ns': step_stamps,
        'step_pose': step_poses,
        'step_pose_ok': pose_ok,
        'step_pose_gap_ns': pose_gaps,
        'reference_stamp_ns': step_stamps[indices],
        'actions': actions,
        'audio': np.stack([o.audio for o in observations]),
        'audio_stamps_ns': np.stack([o.audio_stamps_ns for o in observations]),
        'image_stamps_ns': np.stack([o.image_stamps_ns for o in observations]),
        'pose_current': np.stack([o.pose_current for o in observations]),
        'pose_previous': np.stack([o.pose_previous for o in observations]),
        'pose_gap_ns': np.array([o.pose_gap_ns for o in observations], dtype=np.int64),
        'gripper': gripper,
        # Provenance, so a dataset can never be read without knowing how the
        # numbers in it were defined.
        'chunk_length': np.array(args.chunk_length),
        'action_frame': np.array(args.frame),
        'action_source': np.array(action_source),
        'gripper_source': np.array(gripper_source),
        'pose_source': np.array(f'tf:{args.base_frame}->{args.tool_frame}'),
        'audio_topic': np.array(args.audio_topic),
        'pose_dt_sec': np.array(spec.pose_dt_sec),
        'audio_history_sec': np.array(spec.audio_history_sec),
        'n_images': np.array(spec.n_images),
    }
    if commanded_actions is not None:
        payload['commanded_actions'] = np.stack(
            [chunker.chunk(commanded_actions, gripper, int(k)) for k in indices])
    if not args.no_images:
        payload['images'] = np.stack([o.images for o in observations])

    np.savez_compressed(args.out, **payload)
    moved = np.linalg.norm(payload['actions'][:, :, :3], axis=-1)
    print(f'action chunks      : {payload["actions"].shape} '
          f'({args.chunk_length} steps x 10, frame={args.frame})')
    print(f'per-step motion    : median {np.median(moved) * 1e3:.3f} mm, '
          f'max {moved.max() * 1e3:.3f} mm')
    print(f'pose gap           : median {np.median(payload["pose_gap_ns"]) / 1e6:.2f} ms')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
