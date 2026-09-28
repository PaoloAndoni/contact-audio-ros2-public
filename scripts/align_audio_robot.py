"""
Pair every audio window in a bag with the robot state that produced it.

This is the payoff for the timestamp discipline: because a Spectrogram's
header.stamp is the capture time of its FIRST sample and is never re-stamped,
the robot pose at that instant can be looked up after the fact.

A window spans a whole second, and the arm moves during it, so a single pose is
not the whole truth. Both ends are recorded -- the pose at the window's start
and at its end -- because the motion across the window cannot be recovered later
if only one is kept.

Alignment is done OFFLINE from a bag rather than live, so it can be re-run with
different strategies against the same recording. Recording itself is plain
`ros2 bag record`, which is far better tested than anything custom.

Usage:
    ros2 bag record -s mcap -o run1 /audio/spectrogram /audio/contact_prediction \\
        /joint_states /tf
    python3 scripts/align_audio_robot.py run1 --out run1_aligned.npz
"""
import argparse
import sys

import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

AUDIO_TOPIC = '/audio/spectrogram'
JOINT_TOPIC = '/joint_states'


def read_bag(path, topics):
    """Yield (topic, deserialised message, bag timestamp) for the given topics."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id=''),
        rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                    output_serialization_format='cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    if missing:
        raise SystemExit(f'{path}: bag does not contain {missing}. It has: '
                         f'{sorted(types)}')
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        yield topic, deserialize_message(raw, get_message(types[topic])), stamp


def stamp_ns(header):
    """Return a header stamp in integer nanoseconds."""
    return header.stamp.sec * 10**9 + header.stamp.nanosec


def interpolate(times_ns, values, target_ns):
    """
    Linearly interpolate joint positions at target_ns.

    Returns (positions, gap_ns) where gap_ns is the distance to the nearest
    sample -- reported so a caller can tell interpolation from extrapolation
    across a hole in the data.
    """
    if target_ns < times_ns[0] or target_ns > times_ns[-1]:
        return None, None
    idx = int(np.searchsorted(times_ns, target_ns))
    if idx == 0:
        return values[0], 0
    before, after = times_ns[idx - 1], times_ns[min(idx, len(times_ns) - 1)]
    span = after - before
    frac = 0.0 if span == 0 else (target_ns - before) / span
    interp = values[idx - 1] + frac * (values[min(idx, len(values) - 1)] - values[idx - 1])
    return interp, int(min(target_ns - before, after - target_ns))


def main():
    """Align the bag and write the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag')
    parser.add_argument('--out', default='aligned.npz')
    parser.add_argument('--audio-topic', default=AUDIO_TOPIC)
    parser.add_argument('--joint-topic', default=JOINT_TOPIC)
    parser.add_argument('--with-features', action='store_true',
                        help='also store every 98x128 filterbank (large)')
    parser.add_argument('--max-gap-ms', type=float, default=50.0,
                        help='reject a window if the nearest joint sample is further')
    args = parser.parse_args()

    audio, joint_times, joint_values, joint_names = [], [], [], None
    for topic, msg, _ in read_bag(args.bag, (args.audio_topic, args.joint_topic)):
        if topic == args.joint_topic:
            joint_times.append(stamp_ns(msg.header))
            joint_values.append(list(msg.position))
            joint_names = joint_names or list(msg.name)
        else:
            audio.append(msg)

    if not audio:
        raise SystemExit(f'no messages on {args.audio_topic}')
    if len(joint_times) < 2:
        raise SystemExit(f'need at least two {args.joint_topic} messages')

    order = np.argsort(np.array(joint_times, dtype=np.int64))
    times_ns = np.array(joint_times, dtype=np.int64)[order]
    values = np.array(joint_values, dtype=np.float64)[order]
    print(f'audio windows      : {len(audio)}')
    print(f'joint samples      : {len(times_ns)} ({joint_names})')
    print(f'joint rate         : '
          f'{1e9 * (len(times_ns) - 1) / max(1, times_ns[-1] - times_ns[0]):.1f} Hz')

    rows, skipped_range, skipped_gap, gaps = [], 0, 0, []
    for msg in audio:
        start = stamp_ns(msg.header)
        end = start + int(round(float(msg.window_duration_sec) * 1e9))
        q_start, gap_a = interpolate(times_ns, values, start)
        q_end, gap_b = interpolate(times_ns, values, end)
        if q_start is None or q_end is None:
            skipped_range += 1
            continue
        gap = max(gap_a, gap_b)
        if gap > args.max_gap_ms * 1e6:
            skipped_gap += 1
            continue
        gaps.append(gap)
        rows.append((start, end, int(msg.start_sample_index), q_start, q_end,
                     np.asarray(msg.data, dtype=np.float32) if args.with_features
                     else None))

    if not rows:
        print('RESULT: FAIL - nothing could be aligned', file=sys.stderr)
        raise SystemExit(1)

    gaps_ms = np.array(gaps) / 1e6
    print(f'aligned            : {len(rows)}')
    print(f'skipped (no cover) : {skipped_range}')
    print(f'skipped (gap)      : {skipped_gap}')
    print(f'nearest-sample gap : median {np.median(gaps_ms):.2f} ms, '
          f'max {gaps_ms.max():.2f} ms')

    payload = {
        'window_start_ns': np.array([r[0] for r in rows], dtype=np.int64),
        'window_end_ns': np.array([r[1] for r in rows], dtype=np.int64),
        'start_sample_index': np.array([r[2] for r in rows], dtype=np.int64),
        'joint_positions_start': np.array([r[3] for r in rows], dtype=np.float64),
        'joint_positions_end': np.array([r[4] for r in rows], dtype=np.float64),
        'joint_names': np.array(joint_names),
        'nearest_gap_ns': np.array(gaps, dtype=np.int64),
    }
    if args.with_features:
        payload['features'] = np.stack([r[5] for r in rows])
        payload['n_frames'] = np.array(audio[0].n_frames)
        payload['n_mels'] = np.array(audio[0].n_mels)
        payload['preprocessing_id'] = np.array(audio[0].preprocessing_id)
    np.savez(args.out, **payload)

    moved = np.abs(payload['joint_positions_end'] - payload['joint_positions_start'])
    print(f'joint motion within a window: median {np.median(moved):.4f} rad, '
          f'max {moved.max():.4f} rad')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
