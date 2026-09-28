"""
Check assembled observations against ground truth, and prove the check can fail.

Three things are verified, each against something external to the assembler:

  CAMERA  every frame is DECODED. fake_camera.py writes each frame's sequence
          number and stamp into its own first pixels, so this checks that the
          image bytes belonging to a given stamp are the ones that stamp names
          -- not merely that two frames of the right shape were returned. An
          assembler that returned the same frame twice, or the two oldest
          frames, would satisfy every shape and ordering check ever written.
  AUDIO   causality and span: nothing after the reference, and the history
          really reaches back audio_history_sec.
  POSE    against fake_robot_state.py's closed form, the same way
          verify_alignment.py checks joint positions -- plus the pose PAIR
          spacing, which must be exactly pose_dt and not "the last two
          messages".

SENSITIVITY: the pose comparison is repeated with the reference shifted 100 ms,
and the frame selection is re-run one frame later. Both errors must grow.

Usage: python3 scripts/verify_observation.py run1_actions.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'src',
    'contact_audio_perception'))
from fake_camera import decode_frame  # noqa: E402
from fake_robot_state import tool_pose  # noqa: E402

TRANSLATION_TOLERANCE_M = 5e-5
QUATERNION_TOLERANCE = 5e-4
SHIFT_MS = 100.0
MIN_GROWTH = 20.0


def main():
    """Verify an assembled observation set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', default='actions.npz')
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    reference = data['reference_stamp_ns']
    image_stamps = data['image_stamps_ns']
    audio_stamps = data['audio_stamps_ns']
    pose_current = data['pose_current']
    pose_previous = data['pose_previous']
    pose_dt = float(data['pose_dt_sec'])
    history = float(data['audio_history_sec'])

    print(f'dataset            : {args.dataset}')
    print(f'observations       : {len(reference)}')
    print(f'images per obs     : {image_stamps.shape[1]}')
    print(f'audio per obs      : {audio_stamps.shape[1]} over {history:g} s')
    print(f'pose pair spacing  : {pose_dt * 1e3:.2f} ms')

    # ---- camera: decode the pixels ----
    print()
    print('=== camera: frame identity decoded from pixel content ===')
    if 'images' not in data:
        print('dataset was written with --no-images; camera identity NOT checked')
        decoded_ok = False
        mismatches = -1
    else:
        images = data['images']
        mismatches, unreadable = 0, 0
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                try:
                    _, encoded_ns = decode_frame(images[i, j])
                except ValueError:
                    unreadable += 1
                    continue
                if encoded_ns != int(image_stamps[i, j]):
                    mismatches += 1
        total = images.shape[0] * images.shape[1]
        print(f'frames decoded     : {total - unreadable}/{total}')
        print(f'stamp mismatches   : {mismatches} '
              f'(the pixels must agree with the stamp they were filed under)')
        decoded_ok = unreadable == 0 and mismatches == 0

        # Sensitivity: the SAME comparison against the neighbouring frame must
        # fail. If it did not, the decode is not discriminating between frames.
        wrong = 0
        for i in range(images.shape[0]):
            try:
                _, encoded_ns = decode_frame(images[i, -1])
            except ValueError:
                continue
            if encoded_ns == int(image_stamps[i, -2]):
                wrong += 1
        print(f'control            : newest frame matched the PREVIOUS stamp '
              f'{wrong}/{images.shape[0]} times (must be 0, else the check is blind)')
        decoded_ok = decoded_ok and wrong == 0

    # ---- causality ----
    print()
    print('=== causality: nothing from after the reference ===')
    image_future = int((image_stamps > reference[:, None]).sum())
    audio_future = int((audio_stamps > reference[:, None]).sum())
    print(f'camera frames after the reference : {image_future} (must be 0)')
    print(f'audio items after the reference   : {audio_future} (must be 0)')
    newest_age = (reference - image_stamps[:, -1]) / 1e6
    print(f'newest frame age   : median {np.median(newest_age):.1f} ms, '
          f'max {newest_age.max():.1f} ms')

    ordered = bool(np.all(np.diff(image_stamps, axis=1) > 0) and
                   np.all(np.diff(audio_stamps, axis=1) > 0))
    span = (reference - audio_stamps[:, 0]) / 1e9
    print(f'audio history span : median {np.median(span):.3f} s '
          f'(configured {history:g} s)')

    # ---- pose against the closed form ----
    print()
    print('=== pose: against the analytic motion ===')
    truth_current = np.array([tool_pose(t / 1e9) for t in reference])
    truth_previous = np.array(
        [tool_pose((t - pose_dt * 1e9) / 1e9) for t in reference])
    err_current = np.abs(pose_current - truth_current)
    err_previous = np.abs(pose_previous - truth_previous)
    worst_t = max(err_current[:, :3].max(), err_previous[:, :3].max())
    worst_q = max(err_current[:, 3:].max(), err_previous[:, 3:].max())
    print(f'pose_current/previous error : {worst_t:.3e} m translation, '
          f'{worst_q:.3e} quaternion')

    # The pair must be a fixed interval apart. With a 100 Hz pose stream, "the
    # last two messages" would be 10 ms apart instead of pose_dt -- a factor of
    # nearly seven at 15 Hz, and invisible in any shape or smoothness check.
    measured = np.linalg.norm(pose_current[:, :3] - pose_previous[:, :3], axis=1)
    expected = np.linalg.norm(truth_current[:, :3] - truth_previous[:, :3], axis=1)
    spacing_error = np.abs(measured - expected).max()
    print(f'pair separation error       : {spacing_error:.3e} m '
          f'(median motion {np.median(expected) * 1e3:.3f} mm per {pose_dt * 1e3:.1f} ms)')

    # ---- sensitivity ----
    print()
    print(f'=== sensitivity: reference shifted +{SHIFT_MS:g} ms ===')
    bumped = np.array([tool_pose((t + SHIFT_MS * 1e6) / 1e9) for t in reference])
    shifted_t = np.abs(pose_current[:, :3] - bumped[:, :3]).max()
    shifted_q = np.abs(pose_current[:, 3:] - bumped[:, 3:]).max()
    print(f'translation        : {shifted_t:.3e} m '
          f'({shifted_t / max(worst_t, 1e-18):.0f}x the aligned error)')
    print(f'quaternion         : {shifted_q:.3e} '
          f'({shifted_q / max(worst_q, 1e-18):.0f}x)')

    checks = [
        ('camera frame identity decoded and correct', decoded_ok),
        ('no camera frame comes from after the reference', image_future == 0),
        ('no audio item comes from after the reference', audio_future == 0),
        ('streams are ordered oldest first', ordered),
        ('audio history covers the configured span',
         abs(float(np.median(span)) - history) < 0.2 * history),
        ('pose matches ground truth', worst_t < TRANSLATION_TOLERANCE_M and
         worst_q < QUATERNION_TOLERANCE),
        ('pose pair is exactly pose_dt apart',
         spacing_error < TRANSLATION_TOLERANCE_M),
        ('shifting the reference is caught',
         min(shifted_t / max(worst_t, 1e-18),
             shifted_q / max(worst_q, 1e-18)) > MIN_GROWTH),
    ]
    print()
    for name, ok in checks:
        print(f'[{"PASS" if ok else "FAIL"}] {name}')
    failed = [n for n, ok in checks if not ok]
    print()
    print('RESULT:', 'OBSERVATIONS VERIFIED' if not failed
          else f'FAIL ({", ".join(failed)})')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
