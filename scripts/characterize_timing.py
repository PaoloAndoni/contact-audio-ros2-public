"""
Timestamp characterisation gate: run before recording any synchronised bag.

Subscribes to /audio/spectrogram (and /audio/contact_prediction) while the DAQ
writes a per-callback timing CSV, then decides whether the derived capture
stamps really describe when the sound happened.

Criteria are reported as PASS, FAIL or N/A. A criterion that cannot be evaluated
is N/A -- never a silent PASS. In particular, missing ADC timestamps and absent
PortAudio status flags are separate checks, because a backend that supplies no
ADC time at all would otherwise look identical to a clean one.

  --source mic   every criterion applies; this is the acceptance run
  --source wav   the control: no hardware clock exists, so the ADC, status and
                 drift criteria report N/A rather than pretending to pass

Thresholds are CLI arguments so the lab machine can be held to the tolerance the
robot synchronisation actually requires. Do NOT relax them to make a known-bad
backend pass.

Usage:
  python3 scripts/characterize_timing.py 70 --source mic --csv /tmp/daq_timing.csv
  python3 scripts/characterize_timing.py 45 --source wav
"""
import argparse
import sys
import time

from contact_audio_interfaces.msg import ContactPrediction, Spectrogram
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

PASS, FAIL, NA = 'PASS', 'FAIL', 'N/A'


class Listener(Node):
    """Record arrival time and capture stamp for every published message."""

    def __init__(self):
        """Subscribe to the spectrogram and prediction topics."""
        super().__init__('characterize_timing')
        reliable = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        best = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)
        self.windows = []
        self.predictions = []
        self.buffer_sec = None
        self.rate = None
        self.create_subscription(Spectrogram, '/audio/spectrogram', self.on_spec, best)
        self.create_subscription(ContactPrediction, '/audio/contact_prediction',
                                 self.on_pred, reliable)

    def on_spec(self, msg):
        """Record one spectrogram's stamp, index and arrival."""
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self.buffer_sec = float(msg.window_duration_sec)
        self.rate = int(msg.source_sample_rate)
        self.windows.append((stamp, int(msg.start_sample_index),
                             self.get_clock().now().nanoseconds))

    def on_pred(self, msg):
        """Record one prediction's stamp and arrival."""
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self.predictions.append((stamp, self.get_clock().now().nanoseconds))


def read_timing_csv(path):
    """Parse the DAQ's per-callback timing log, or return None if absent."""
    rows = []
    try:
        with open(path) as handle:
            if not handle.readline():
                return rows
            for line in handle:
                parts = line.rstrip('\n').split(',')
                if len(parts) < 7:
                    continue
                rows.append({
                    'callback_ns': int(parts[0]), 'adc': float(parts[1]),
                    'current': float(parts[2]), 'index': int(parts[3]),
                    'first_ns': int(parts[4]), 'frames': int(parts[5]),
                    'status': parts[6],
                    'residual_ms': float(parts[7]) if len(parts) > 7 else 0.0,
                })
    except FileNotFoundError:
        return None
    return rows


def collect(seconds):
    """Spin for the requested duration and return what was heard."""
    rclpy.init()
    node = Listener()
    deadline = time.time() + seconds
    while time.time() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    out = (node.windows, node.predictions, node.buffer_sec, node.rate)
    node.destroy_node()
    rclpy.shutdown()
    return out


def main():
    """Run the gate and exit non-zero unless every applicable criterion passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seconds', nargs='?', type=float, default=70.0)
    parser.add_argument('--source', choices=('mic', 'wav'), default='mic')
    parser.add_argument('--csv', default='/tmp/daq_timing.csv')
    parser.add_argument('--require-predictions', action='store_true')
    parser.add_argument('--max-drift-ppm', type=float, default=200.0,
                        help='200 ppm is 12 ms of slip per minute of recording')
    parser.add_argument('--max-discontinuity-ms', type=float, default=5.0)
    parser.add_argument('--max-discontinuities', type=int, default=0)
    parser.add_argument('--min-callbacks', type=int, default=100)
    parser.add_argument('--stamp-tolerance-ms', type=float, default=2.0)
    parser.add_argument('--age-slack-ms', type=float, default=400.0)
    args = parser.parse_args()

    windows, predictions, buffer_sec, rate = collect(args.seconds)
    buffer_sec = buffer_sec or 1.0
    rate = rate or 44100
    is_mic = args.source == 'mic'
    checks = []

    def check(name, status, detail):
        checks.append((name, status, detail))

    print(f'source      : {args.source}')
    print(f'windows     : {len(windows)}')
    print(f'predictions : {len(predictions)}')
    if len(windows) < 10:
        print('RESULT: FAIL - too few windows to characterise')
        raise SystemExit(1)

    stamps = np.array([w[0] for w in windows], dtype=np.int64)
    indices = np.array([w[1] for w in windows], dtype=np.int64)
    arrivals = np.array([w[2] for w in windows], dtype=np.int64)
    ages_ms = (arrivals - stamps) / 1e6

    future = int(np.sum(ages_ms < 0.0))
    check('no future window stamps', PASS if future == 0 else FAIL,
          f'{future}/{len(ages_ms)} in the future (min age {ages_ms.min():.0f} ms)')

    steps = np.diff(stamps)
    check('stamps strictly monotonic', PASS if np.all(steps > 0) else FAIL,
          f'{int(np.sum(steps <= 0))} non-increasing steps')

    expected_ms = np.diff(indices) / rate * 1000.0
    residual = np.abs(steps / 1e6 - expected_ms)
    worst = float(residual.max()) if residual.size else 0.0
    check('stamp deltas match sample-index deltas',
          PASS if worst < args.stamp_tolerance_ms else FAIL,
          f'worst |d| {worst:.3f} ms (tolerance {args.stamp_tolerance_ms:g} ms)')

    target = buffer_sec * 1000.0
    median_age = float(np.median(ages_ms))
    check('window age near buffer duration',
          PASS if abs(median_age - target) < args.age_slack_ms else FAIL,
          f'median {median_age:.0f} ms vs {target:.0f} ms expected')

    if predictions:
        p_ages = np.array([(a - s) / 1e6 for s, a in predictions])
        ok = bool(np.all(p_ages > 0)) and np.median(p_ages) >= target - args.age_slack_ms
        check('prediction age sane', PASS if ok else FAIL,
              f'median {np.median(p_ages):.0f} ms, min {p_ages.min():.0f} ms')
    elif args.require_predictions:
        check('prediction age sane', FAIL, 'no predictions received')
    else:
        check('prediction age sane', NA, 'no predictions running')

    rows = read_timing_csv(args.csv) if is_mic else None
    if not is_mic:
        for name in ('timing log present', 'ADC timestamps available',
                     'no PortAudio status events', 'sample-clock drift bounded',
                     'no capture discontinuities'):
            check(name, NA, 'file replay has no capture hardware')
    elif rows is None:
        check('timing log present', FAIL, f'{args.csv} does not exist')
        for name in ('ADC timestamps available', 'no PortAudio status events',
                     'sample-clock drift bounded', 'no capture discontinuities'):
            check(name, FAIL, 'cannot evaluate without the timing log')
    elif len(rows) < args.min_callbacks:
        check('timing log present', FAIL,
              f'{len(rows)} callbacks, need {args.min_callbacks}')
        for name in ('ADC timestamps available', 'no PortAudio status events',
                     'sample-clock drift bounded', 'no capture discontinuities'):
            check(name, FAIL, 'too few callbacks to evaluate')
    else:
        check('timing log present', PASS, f'{len(rows)} callbacks')

        missing = [r for r in rows if r['adc'] <= 0.0]
        check('ADC timestamps available', PASS if not missing else FAIL,
              f'{len(missing)}/{len(rows)} callbacks without inputBufferAdcTime')

        flagged = [r for r in rows if r['status']]
        check('no PortAudio status events', PASS if not flagged else FAIL,
              f'{len(flagged)} status events')

        adc = np.array([r['adc'] for r in rows])
        frames = float(np.median([r['frames'] for r in rows]))
        if missing:
            check('sample-clock drift bounded', FAIL,
                  'ADC time unavailable, drift cannot be measured')
            check('no capture discontinuities', FAIL,
                  'ADC time unavailable, discontinuities cannot be measured')
        else:
            # Robust rate estimate: the median callback interval versus the
            # interval the sample counter implies. A least-squares slope is
            # skewed by the jumps this is meant to detect.
            median_step = float(np.median(np.diff(adc)))
            implied = frames / median_step if median_step > 0 else 0.0
            drift_ppm = (implied - rate) / rate * 1e6
            check('sample-clock drift bounded',
                  PASS if abs(drift_ppm) <= args.max_drift_ppm else FAIL,
                  f'{drift_ppm:+.0f} ppm (implied {implied:.0f} Hz vs {rate} Hz '
                  f'nominal, limit +/-{args.max_drift_ppm:g})')

            residual_s = (adc - adc[0]) - (np.array([r['index'] for r in rows])
                                           - rows[0]['index']) / rate
            jumps = np.abs(np.diff(residual_s)) * 1000.0
            n_jumps = int(np.sum(jumps > args.max_discontinuity_ms))
            check('no capture discontinuities',
                  PASS if n_jumps <= args.max_discontinuities else FAIL,
                  f'{n_jumps} jumps over {args.max_discontinuity_ms:g} ms '
                  f'(worst {jumps.max():.0f} ms, limit {args.max_discontinuities})')

    print()
    for name, status, detail in checks:
        print(f'[{status:^4}] {name:<42} {detail}')
    failed = [name for name, status, _ in checks if status == FAIL]
    print()
    if failed:
        print(f'RESULT: TIMING GATE FAIL ({len(failed)} criteria: {", ".join(failed)})')
    else:
        na = sum(1 for _, status, _ in checks if status == NA)
        print(f'RESULT: TIMING GATE PASS'
              + (f' ({na} criteria not applicable to {args.source})' if na else ''))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
