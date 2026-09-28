"""
Check an aligned dataset against ground truth, not against plausibility.

fake_robot_state.py moves the arm as a known function of absolute time, so for
any window timestamp the exact joint positions can be recomputed. This compares
the aligned values against that closed form.

That distinction matters: an alignment that is offset by a fixed delay still
produces smooth, believable-looking poses. Only comparison with ground truth
catches it. The deliberate check is TIME SHIFT SENSITIVITY -- the same
comparison is repeated with the timestamps deliberately shifted, and the error
must grow. If it does not, the test is not actually measuring alignment.

Usage: python3 scripts/verify_alignment.py aligned.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_robot_state import joint_positions  # noqa: E402

# Interpolating a sinusoid from 100 Hz samples has a bounded chord error; a few
# milliradians is the honest bar. Anything near zero would mean the fake robot
# and the verifier share a bug rather than that alignment works.
TOLERANCE_RAD = 5e-3
SHIFT_MS = 100.0


def expected(times_ns):
    """Return the exact joint positions at each timestamp."""
    return np.array([joint_positions(t / 1e9) for t in times_ns], dtype=np.float64)


def main():
    """Compare the aligned dataset with the analytic motion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', default='aligned.npz')
    parser.add_argument('--tolerance', type=float, default=TOLERANCE_RAD)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    start_ns = data['window_start_ns']
    end_ns = data['window_end_ns']
    got_start = data['joint_positions_start']
    got_end = data['joint_positions_end']
    print(f'windows            : {len(start_ns)}')
    print(f'joints             : {list(data["joint_names"])}')

    err_start = np.abs(got_start - expected(start_ns))
    err_end = np.abs(got_end - expected(end_ns))
    worst = max(err_start.max(), err_end.max())
    median = np.median(np.concatenate([err_start.ravel(), err_end.ravel()]))
    print(f'error vs truth     : median {median:.6f} rad, worst {worst:.6f} rad '
          f'(tolerance {args.tolerance:g})')

    per_joint = err_start.max(axis=0)
    for name, e in zip(data['joint_names'], per_joint):
        print(f'   {str(name):<22} worst {e:.6f} rad')

    # Sanity: shifting the timestamps must make the error worse. Without this,
    # a check that trivially passes would look identical to one that works.
    shifted = np.abs(got_start - expected(start_ns + int(SHIFT_MS * 1e6)))
    print(f'error at +{SHIFT_MS:g} ms shift : worst {shifted.max():.6f} rad '
          f'(must exceed the aligned error)')

    duration_s = (end_ns - start_ns) / 1e9
    print(f'window duration    : median {np.median(duration_s):.4f} s')

    checks = [
        ('alignment matches ground truth', worst < args.tolerance),
        ('shifting time makes it worse', shifted.max() > worst * 5),
        ('window duration is ~1 s', abs(np.median(duration_s) - 1.0) < 0.01),
        ('timestamps strictly increasing', bool(np.all(np.diff(start_ns) > 0))),
    ]
    print()
    for name, ok in checks:
        print(f'[{"PASS" if ok else "FAIL"}] {name}')
    failed = [n for n, ok in checks if not ok]
    print()
    print('RESULT:', 'ALIGNMENT VERIFIED' if not failed
          else f'FAIL ({", ".join(failed)})')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
