"""Gate 4: compare the two-node and four-node pipelines over ROS.

Runs each pipeline over the same WAV and matches their ContactPrediction
messages by header.stamp, which both pipelines set to the capture time of the
buffer's first sample. Only windows that both pipelines actually produced are
compared -- the new pipeline drops stale spectrograms by design, so it is
expected to emit fewer.

Usage: python3 scripts/verify_block2_ros.py <new.jsonl> <legacy.jsonl>
       python3 scripts/verify_block2_ros.py --record <out.jsonl> <seconds>
"""
import json
import sys
import time

from contact_audio_interfaces.msg import ContactPrediction
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class Recorder(Node):
    """Write every ContactPrediction to a JSONL file keyed by capture stamp."""

    def __init__(self, path):
        """Subscribe to the prediction topic."""
        super().__init__('verify_block2_recorder')
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        # NB: not self.handle -- rclpy.Node.handle is a read-only property.
        self._out = open(path, 'w')
        self.count = 0
        self.create_subscription(
            ContactPrediction, '/audio/contact_prediction', self.on_pred, qos)

    def on_pred(self, msg):
        """Append one prediction."""
        self._out.write(json.dumps({
            'index': int(msg.start_sample_index),
            'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'label': msg.label,
            # Not rounded: this file backs a claim of exact equality, and
            # rounding to 6 places would manufacture agreement at 1e-7.
            'confidence': float(msg.confidence),
            'class_names': list(msg.class_names),
            'probabilities': [float(p) for p in msg.probabilities],
        }) + '\n')
        self._out.flush()
        self.count += 1


def record(path, seconds):
    """Collect predictions for a while."""
    rclpy.init()
    node = Recorder(path)
    deadline = time.time() + seconds
    while time.time() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    print(f'recorded {node.count} predictions to {path}')
    node._out.close()
    node.destroy_node()
    rclpy.shutdown()


def load(path):
    """Read a JSONL recording keyed by source-window sample index.

    Keyed on start_sample_index, not header.stamp: each run anchors its clock to
    its own start time, so absolute stamps never match across runs, and both
    pipelines drop a different sparse subset of windows under load.
    """
    out = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            out[int(row['index'])] = row
    return out


MIN_SHARED = 3


def compare(new_path, legacy_path, tol=1e-4):
    """Compare two recordings on the windows they share."""
    new, legacy = load(new_path), load(legacy_path)
    shared = sorted(set(new) & set(legacy))
    print(f'new pipeline    : {len(new)} predictions')
    print(f'legacy pipeline : {len(legacy)} predictions')
    print(f'shared windows  : {len(shared)} (by start_sample_index {shared})')
    if len(shared) < MIN_SHARED:
        # One agreeing window is not evidence of equivalence; it is evidence
        # that one window agreed.
        print(f'RESULT: FAIL - only {len(shared)} shared window(s), need '
              f'{MIN_SHARED}. Run the legacy pipeline longer so it covers more '
              f'of the new pipeline''s window indices.')
        return False

    label_ok = order_ok = prob_ok = True
    worst = 0.0
    for index in shared:
        a, b = new[index], legacy[index]
        label_ok &= a['label'] == b['label']
        order_ok &= a['class_names'] == b['class_names']
        delta = max(abs(x - y) for x, y in zip(a['probabilities'], b['probabilities']))
        worst = max(worst, delta)
        prob_ok &= delta < tol
        prob_ok &= abs(a['confidence'] - b['confidence']) < tol

    print(f'labels match    : {label_ok}')
    print(f'class order     : {order_ok}')
    print(f'probabilities   : {prob_ok} (worst |d| {worst:.3e}, tol {tol:g})')
    ok = label_ok and order_ok and prob_ok
    print('RESULT:', 'GATE 4 PASS' if ok else 'GATE 4 FAIL')
    return ok


def main():
    """Record or compare, depending on the arguments."""
    if len(sys.argv) >= 2 and sys.argv[1] == '--record':
        record(sys.argv[2], float(sys.argv[3]))
        return
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if compare(sys.argv[1], sys.argv[2]) else 1)


if __name__ == '__main__':
    main()
