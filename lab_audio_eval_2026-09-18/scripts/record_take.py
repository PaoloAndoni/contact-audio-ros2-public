#!/usr/bin/env python3
"""Record one take into a sqlite3 rosbag in the exact 18 Sept lab format,
then immediately measure it and say whether to keep it.

WHY A BAG AND NOT A WAV: every analysis script already reads /audio/raw_chunk
out of sqlite3 bags. Writing the same format means the new recordings are
drop-in comparable with the lab session -- same loader, same quality pass,
same figures -- with nothing special-cased for "the home ones".

WHY IT SELF-VERIFIES: the 18 Sept session recorded 17 takes and nobody knew
they were unusable until they were analysed days later at home. This prints a
verdict per take, before you move on to the next one.

THE VOLUME RESETS. WSLg puts RDPSource back to 100% when the source suspends,
so the attenuation is applied here, in the same process as the capture, every
single take. Do not rely on having set it earlier.
"""
import argparse
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import sounddevice as sd

import rclpy.serialization
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quality  # noqa: E402

TOPIC = '/audio/raw_chunk'
MSG_TYPE = 'contact_audio_interfaces/msg/AudioChunk'
CHUNK_FRAMES = 2048          # matches the lab bags exactly
RATE = 48000
CHANNELS = 1
DEFAULT_VOLUME = '55%'


def set_volume(vol):
    """Apply the attenuation. WSLg resets this, so it is done every take."""
    try:
        subprocess.run(['pactl', 'set-source-volume', 'RDPSource', vol],
                       check=True, capture_output=True)
        out = subprocess.run(['pactl', 'get-source-volume', 'RDPSource'],
                             capture_output=True, text=True).stdout
        return out.splitlines()[0].strip() if out else 'unknown'
    except Exception as exc:                                  # noqa: BLE001
        return f'FAILED to set volume: {exc}'


def write_bag(path, audio, rate, channels):
    """Write int16 audio as AudioChunk messages, lab-format sqlite3 bag."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''))
    # Jazzy's TopicMetadata requires an explicit id; Humble's did not.
    writer.create_topic(rosbag2_py.TopicMetadata(
        id=0, name=TOPIC, type=MSG_TYPE, serialization_format='cdr'))

    msg_cls = get_message(MSG_TYPE)
    n_frames = len(audio) // channels
    t0_ns = time.time_ns()
    period_ns = int(1e9 * CHUNK_FRAMES / rate)

    for i, start in enumerate(range(0, n_frames, CHUNK_FRAMES)):
        end = min(start + CHUNK_FRAMES, n_frames)
        m = msg_cls()
        stamp_ns = t0_ns + i * period_ns
        # header.stamp is the time of the FIRST frame in this chunk -- the
        # contract the whole pipeline depends on. Never re-stamp with now().
        m.header.stamp.sec = stamp_ns // 10**9
        m.header.stamp.nanosec = stamp_ns % 10**9
        m.header.frame_id = 'contact_mic'
        m.sample_rate = rate
        m.channels = channels
        m.start_sample_index = start
        m.data = audio[start * channels:end * channels].tolist()
        writer.write(TOPIC, rclpy.serialization.serialize_message(m), stamp_ns)
    del writer
    return (n_frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', help='take name, e.g. table_cleaning_noiseoff_01')
    ap.add_argument('--sec', type=float, default=15.0)
    ap.add_argument('--volume', default=DEFAULT_VOLUME)
    ap.add_argument('--outdir', default='~/home_bags')
    args = ap.parse_args()

    outdir = pathlib.Path(os.path.expanduser(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    bag = outdir / args.name
    if bag.exists():
        print(f'REFUSING: {bag} already exists. Delete it or pick another name.')
        return 2

    print(f'volume: {set_volume(args.volume)}')
    print(f'recording "{args.name}" for {args.sec:.0f}s at {RATE} Hz mono')
    for n in (3, 2, 1):
        print(f'  {n}...', flush=True)
        time.sleep(1)
    print('  GO', flush=True)

    rec = sd.rec(int(args.sec * RATE), samplerate=RATE, channels=CHANNELS,
                 dtype='int16')
    sd.wait()
    audio = np.asarray(rec, dtype=np.int16).reshape(-1)
    print('  done', flush=True)

    n_msgs = write_bag(bag, audio, RATE, CHANNELS)
    size_mb = sum(f.stat().st_size for f in bag.iterdir()) / 2**20
    print(f'  wrote {n_msgs} chunks, {size_mb:.2f} MB -> {bag}')

    # Verify immediately, using the same code the lab evaluation used.
    loaded = {'sample_rate': RATE, 'channels': CHANNELS,
              'duration_s': len(audio) / RATE, 'n_chunks': n_msgs,
              'gaps': [], 'gap_seconds': 0.0, 'overlap_frames': 0,
              'data': audio.reshape(-1, CHANNELS)}
    mono = audio.astype(np.float32) / 32768.0
    q = quality.analyse(loaded, mono)
    verdict, reasons = quality.verdict(q)

    print()
    print('=' * 66)
    print(f'  peak      {int(q["peak_counts"]):6d} counts ({q["peak_dbfs"]:+7.1f} dBFS)')
    print(f'  rms       {q["rms_dbfs"]:+7.1f} dBFS')
    print(f'  floor     {q["floor_dbfs"]:+7.1f} dBFS')
    print(f'  clipped   {q["pct_clipped"]:.3f}%')
    print(f'  events    {q["n_events"]}')
    print(f'  SNR       {q["snr_db"]:.1f} dB')
    print(f'  VERDICT   {verdict}')
    for r in reasons:
        print(f'            - {r}')
    print('=' * 66)
    if verdict == 'UNUSABLE':
        print('  Do not keep this take. Delete the bag and redo it.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
