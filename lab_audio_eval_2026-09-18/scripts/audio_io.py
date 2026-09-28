"""Decode /audio/raw_chunk out of a sqlite3 rosbag2 into a continuous signal.

The lab ran Humble and MCAP storage was unavailable, so every bag from
2026-09-18 is sqlite3. Messages are CDR-serialised AudioChunk.

Reconstruction is done by start_sample_index, not by concatenation: the
message contract says header.stamp is the time of the FIRST frame and
start_sample_index is the per-channel frame index, so laying each chunk
down at its own index is what makes a dropout visible instead of silently
splicing the audio shorter.
"""
import pathlib
import sqlite3

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

AUDIO_TOPIC = '/audio/raw_chunk'


def _db3(bag_dir):
    hits = sorted(pathlib.Path(bag_dir).glob('*.db3'))
    if not hits:
        raise FileNotFoundError(f'no .db3 in {bag_dir}')
    return hits[0]


def read_chunks(bag_dir):
    """Yield (recv_timestamp_ns, AudioChunk) in write order."""
    db = _db3(bag_dir)
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    cur = con.cursor()
    cur.execute('SELECT id, type FROM topics WHERE name=?', (AUDIO_TOPIC,))
    row = cur.fetchone()
    if row is None:
        con.close()
        raise KeyError(f'{AUDIO_TOPIC} not in {bag_dir}')
    tid, ttype = row
    msg_cls = get_message(ttype)
    cur.execute(
        'SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
        (tid,))
    out = [(ts, deserialize_message(bytes(blob), msg_cls))
           for ts, blob in cur.fetchall()]
    con.close()
    return out


def load_bag(bag_dir):
    """Reconstruct one bag's audio. Returns a dict of signal + integrity facts."""
    chunks = read_chunks(bag_dir)
    if not chunks:
        raise ValueError(f'no audio messages in {bag_dir}')

    rates = {c.sample_rate for _, c in chunks}
    chans = {int(c.channels) for _, c in chunks}
    sr = sorted(rates)[0]
    ch = sorted(chans)[0]

    idx0 = chunks[0][1].start_sample_index
    # Frames each chunk claims to start at, and how many it carries.
    starts = np.array([c.start_sample_index - idx0 for _, c in chunks], dtype=np.int64)
    counts = np.array([len(c.data) // max(ch, 1) for _, c in chunks], dtype=np.int64)

    total_frames = int((starts + counts).max())
    buf = np.zeros((total_frames, ch), dtype=np.int16)
    covered = np.zeros(total_frames, dtype=bool)
    for (_, c), s, n in zip(chunks, starts, counts):
        if n == 0:
            continue
        arr = np.asarray(c.data, dtype=np.int16).reshape(n, ch)
        buf[s:s + n] = arr
        covered[s:s + n] = True

    # Gaps: expected next start is previous start + previous count.
    expected = starts[:-1] + counts[:-1]
    delta = starts[1:] - expected
    gap_frames = int(delta[delta > 0].sum())
    overlap_frames = int(-delta[delta < 0].sum())
    gaps = [
        {'after_chunk': int(i), 'frames': int(d), 'seconds': float(d) / sr,
         'at_s': float(expected[i]) / sr}
        for i, d in enumerate(delta) if d > 0
    ]

    # Receive-time vs sample-time: catches a stalled or re-stamped stream.
    recv = np.array([ts for ts, _ in chunks], dtype=np.int64)
    recv_span_s = float(recv[-1] - recv[0]) / 1e9 if len(recv) > 1 else 0.0
    sample_span_s = float(starts[-1]) / sr

    stamps = np.array([c.header.stamp.sec * 10**9 + c.header.stamp.nanosec
                       for _, c in chunks], dtype=np.int64)

    return {
        'name': pathlib.Path(bag_dir).name,
        'sample_rate': int(sr),
        'channels': int(ch),
        'sample_rates_seen': sorted(int(r) for r in rates),
        'channels_seen': sorted(chans),
        'n_chunks': len(chunks),
        'frames': total_frames,
        'duration_s': total_frames / sr,
        'data': buf,                      # (frames, channels) int16
        'covered': covered,               # False where no chunk supplied samples
        'uncovered_frames': int((~covered).sum()),
        'gaps': gaps,
        'gap_frames': gap_frames,
        'gap_seconds': gap_frames / sr,
        'overlap_frames': overlap_frames,
        'recv_span_s': recv_span_s,
        'sample_span_s': sample_span_s,
        'chunk_frames_mode': int(np.bincount(counts).argmax()) if len(counts) else 0,
        'header_stamps_ns': stamps,
        'recv_ns': recv,
    }


def to_mono_float(loaded):
    """Mono float32 in [-1, 1), matching the pipeline's downmix then scale."""
    d = loaded['data'].astype(np.float32)
    mono = d.mean(axis=1) if loaded['channels'] > 1 else d[:, 0]
    return mono / 32768.0
