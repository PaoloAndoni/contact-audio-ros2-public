"""
Check derived actions against the closed form, then prove the check can fail.

THE PROBLEM THIS EXISTS FOR: an off-by-one in the step index produces actions
that are perfectly smooth, physically plausible and completely wrong. Every
naive check passes on them -- they are continuous, they integrate to a sensible
trajectory, their magnitudes are right. Nothing short of comparison with known
truth separates the correct derivation from the displaced one.

fake_robot_state.py moves the tool as a pure function of ABSOLUTE time, so for
any step timestamp the exact pose is recomputable, and therefore so is the
exact action chunk. This compares against that, and then deliberately breaks
the comparison three ways to show the error grows by orders of magnitude:

  1. STEP SHIFT   -- compare against the closed form anchored one step away.
                     The named hazard. It must be caught by a huge margin.
  2. TIME SHIFT   -- compare against the closed form at stamps shifted 100 ms,
                     the same perturbation verify_alignment.py uses.
  3. SMOOTHNESS   -- the control. Measures how smooth the correct actions and
                     the off-by-one actions are, and shows the two are
                     INDISTINGUISHABLE. That is the evidence that a
                     plausibility check would have passed on broken code, and
                     hence that only the closed-form comparison is doing work.

Usage: python3 scripts/verify_actions.py run1_actions.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'src',
    'contact_audio_perception'))
from contact_audio_perception.action import (  # noqa: E402
    ActionChunker,
    channels_grew,
    GRIPPER,
    ROTATION_6D,
    TRANSLATION,
)
from fake_robot_state import tool_pose  # noqa: E402

# The tool is sinusoidal and /tf is sampled at a finite rate, so interpolating
# a pose between two samples has a bounded chord error. At 100 Hz that is a few
# times 1e-6; this is the honest bar. A tolerance near zero would mean the
# verifier and the derivation share an assumption rather than that the
# derivation is right.
TRANSLATION_TOLERANCE_M = 5e-5
ROTATION_TOLERANCE = 5e-4
SHIFT_MS = 100.0
# The sensitivity checks must beat the aligned error by at least this factor.
# Set high on purpose: a check that only just fires is not a check.
MIN_GROWTH = 20.0


def closed_form_poses(stamps_ns):
    """Return the exact tool pose at each timestamp, from the analytic motion."""
    return np.array([tool_pose(t / 1e9) for t in stamps_ns], dtype=np.float64)


def closed_form_actions(stamps_ns, grippers, chunker):
    """Return the exact action chunk for every anchor the chunker allows."""
    return chunker.all_chunks(closed_form_poses(stamps_ns), grippers)


def channel_errors(got, expected):
    """Return (translation, rotation, gripper) worst absolute errors."""
    error = np.abs(got - expected)
    return (error[..., TRANSLATION].max(), error[..., ROTATION_6D].max(),
            error[..., GRIPPER].max())


def smoothness(actions):
    """
    Return the mean magnitude of the second difference along the chunk axis.

    This is what "looks plausible" reduces to: a smooth chunk has a small
    second difference. Reported for the correct and the off-by-one actions so
    the two can be compared directly.
    """
    if actions.shape[1] < 3:
        return float('nan')
    return float(np.abs(np.diff(actions[..., TRANSLATION], n=2, axis=1)).mean())


def main():
    """Compare derived actions with the analytic ground truth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', default='actions.npz')
    parser.add_argument('--translation-tolerance', type=float,
                        default=TRANSLATION_TOLERANCE_M)
    parser.add_argument('--rotation-tolerance', type=float, default=ROTATION_TOLERANCE)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    indices = data['sample_step_index']
    step_stamps = data['step_stamp_ns']
    actions = data['actions']
    grippers = data['gripper']
    chunk_length = int(data['chunk_length'])
    frame = str(data['action_frame'])
    chunker = ActionChunker(chunk_length=chunk_length, frame=frame)

    print(f'dataset            : {args.dataset}')
    print(f'samples            : {len(indices)} of {len(step_stamps)} steps')
    print(f'action chunks      : {actions.shape} (frame={frame})')
    print(f'pose source        : {str(data["pose_source"])}')
    print(f'action source      : {str(data["action_source"])}')
    print(f'gripper source     : {str(data["gripper_source"])}')
    step_ms = np.median(np.diff(step_stamps)) / 1e6
    print(f'step period        : median {step_ms:.2f} ms '
          f'({1000.0 / step_ms:.2f} Hz)')

    # ---- 1. against the closed form, at the timestamps actually used ----
    truth = closed_form_actions(step_stamps, grippers, chunker)[indices]
    t_err, r_err, g_err = channel_errors(actions, truth)
    print()
    print('=== derived vs closed form ===')
    # WORST is the right statistic for the tolerance -- one bad sample is one bad
    # training example. It is the WRONG statistic for asking whether the residual
    # comes from the /tf sample rate, because it is an extreme value set by the
    # widest interpolation gap rather than by the nominal spacing. The median is
    # reported next to it so the two questions stay separable.
    t_per_sample = np.abs(actions - truth)[..., TRANSLATION].max(axis=(1, 2))
    t_median = float(np.median(t_per_sample))
    print(f'translation error  : worst {t_err:.3e} m   '
          f'(tolerance {args.translation_tolerance:g})')
    print(f'translation median : {t_median:.3e} m')
    print(f'rotation 6-D error : worst {r_err:.3e}     '
          f'(tolerance {args.rotation_tolerance:g})')
    print(f'gripper error      : worst {g_err:.3e}')

    # Poses too, not only the deltas: a systematic pose offset cancels out of a
    # difference, so checking actions alone would miss it entirely.
    pose_truth = closed_form_poses(data['reference_stamp_ns'])
    pose_err = np.abs(data['pose_current'] - pose_truth)
    print(f'pose_current error : worst {pose_err[:, :3].max():.3e} m translation, '
          f'{pose_err[:, 3:].max():.3e} quaternion')
    # The cleanest probe of "is the residual just sampling?": ONE interpolated
    # pose, not a chunk. An action aggregates 17 poses, so a single degraded one
    # anywhere in the chunk lands in the action -- which makes the action's
    # median inherit the dropout tail and improve only ~3x with the rate, while
    # the pose's own median improves ~35x, as linear interpolation predicts.
    pose_median = float(np.median(pose_err[:, :3].max(axis=1)))
    print(f'pose_current median: {pose_median:.3e} m translation')

    # The gap to the nearest /tf sample explains the tail. A gap far wider than
    # the nominal spacing means transforms were DROPPED, not merely sampled
    # coarsely -- and a dropped /tf is exactly what will quietly widen the
    # residual on lab data, where nobody knows the closed form to check against.
    gap_ms = data['pose_gap_ns'] / 1e6
    print(f'/tf gap to nearest : median {np.median(gap_ms):.3f} ms, '
          f'worst {gap_ms.max():.3f} ms')

    # ---- 2. SENSITIVITY: shift the step index by one ----
    print()
    print('=== sensitivity 1: step index shifted by one ===')
    shifted_errors = {}
    for offset in (-1, +1):
        moved = indices + offset
        keep = (moved >= 0) & (moved < truth.shape[0])
        if not keep.any():
            continue
        wrong = closed_form_actions(step_stamps, grippers, chunker)[moved[keep]]
        st, sr, _ = channel_errors(actions[keep], wrong)
        shifted_errors[offset] = (st, sr)
        print(f'offset {offset:+d}           : translation {st:.3e} m '
              f'({st / max(t_err, 1e-18):.0f}x the aligned error), '
              f'rotation {sr:.3e} ({sr / max(r_err, 1e-18):.0f}x)')

    # ---- 3. SENSITIVITY: shift every timestamp ----
    print()
    print(f'=== sensitivity 2: timestamps shifted +{SHIFT_MS:g} ms ===')
    bumped = closed_form_actions(
        step_stamps + int(SHIFT_MS * 1e6), grippers, chunker)[indices]
    bt, br, _ = channel_errors(actions, bumped)
    print(f'translation        : {bt:.3e} m ({bt / max(t_err, 1e-18):.0f}x)')
    print(f'rotation 6-D       : {br:.3e} ({br / max(r_err, 1e-18):.0f}x)')

    # ---- 4. CONTROL: smoothness cannot tell them apart ----
    print()
    print('=== control: would a plausibility check have caught it? ===')
    correct_smooth = smoothness(actions)
    offset = +1 if +1 in shifted_errors else -1
    moved = indices + offset
    keep = (moved >= 0) & (moved < truth.shape[0])
    wrong_actions = closed_form_actions(step_stamps, grippers, chunker)[moved[keep]]
    wrong_smooth = smoothness(wrong_actions)
    ratio = wrong_smooth / correct_smooth if correct_smooth else float('inf')
    print(f'mean |2nd difference| of correct actions   : {correct_smooth:.3e} m')
    print(f'mean |2nd difference| of off-by-one actions: {wrong_smooth:.3e} m')
    print(f'ratio                                      : {ratio:.4f}')
    print('Both are equally smooth, so smoothness proves nothing. Only the')
    print('closed-form comparison above separates them.')

    # COMPARE LIKE WITH LIKE. This used to read
    #     min(bt, br) > max(t_err, r_err) * MIN_GROWTH
    # which pits metres against the dimensionless 6-D rotation. The rotation
    # baseline runs several times the translation one, so max() raised the bar
    # by that ratio while min() picked translation to clear it, and the check
    # reported FAIL even though BOTH channels had grown 66-124x -- far past the
    # 20x it exists to demand. channels_grew() holds each to its own baseline.
    aligned = (t_err, r_err)

    def grew(shift_t, shift_r):
        """Say whether both driven channels beat their own error by MIN_GROWTH."""
        return channels_grew((shift_t, shift_r), aligned, MIN_GROWTH)

    step_caught = bool(shifted_errors) and all(
        grew(st, sr) for st, sr in shifted_errors.values())
    checks = [
        ('translation matches ground truth', t_err < args.translation_tolerance),
        ('rotation matches ground truth', r_err < args.rotation_tolerance),
        ('pose_current matches ground truth',
         pose_err[:, :3].max() < args.translation_tolerance),
        ('a one-step shift is caught', step_caught),
        (f'a {SHIFT_MS:g} ms time shift is caught', grew(bt, br)),
        ('smoothness cannot tell them apart (so it is not what passed)',
         0.5 < ratio < 2.0),
        ('every sample has a chunk of the declared length',
         actions.shape[1] == chunk_length and actions.shape[2] == 10),
        ('step timestamps strictly increasing', bool(np.all(np.diff(step_stamps) > 0))),
    ]
    print()
    for name, ok in checks:
        print(f'[{"PASS" if ok else "FAIL"}] {name}')
    failed = [n for n, ok in checks if not ok]
    print()
    print('RESULT:', 'ACTIONS VERIFIED' if not failed else f'FAIL ({", ".join(failed)})')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
