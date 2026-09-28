#!/usr/bin/env python3
"""Step 3: open the ManiWAV `wipe` replay buffer read-only and report what is in it.

Read-only and lazy. The point of the plan is never to pull the RGB array into
RAM, so this touches array METADATA only -- plus a couple of tiny slices to
check dtypes before the big pass commits to them.

Two things this has to survive:
  - camera0_rgb is compressed with imagecodecs_jpegxl, which numcodecs only
    knows about after an explicit register_codecs(). Without it, even reading
    that array's metadata raises.
  - zarr's tree() instantiates every array eagerly, so one unreadable array
    kills the whole listing. Each array is opened defensively instead.
"""
import os
import pathlib
import traceback

import numpy as np
import zarr

try:
    import imagecodecs.numcodecs as inc
    inc.register_codecs()
    CODECS = 'imagecodecs registered'
except Exception as exc:                                  # noqa: BLE001
    CODECS = f'imagecodecs unavailable ({exc.__class__.__name__})'

from paths import ZIP


def human(n):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.1f} {u}'
        n /= 1024
    return f'{n:.1f} PB'


def walk(group, prefix=''):
    """Yield (path, array_or_None, error) without letting one bad array abort."""
    for key in sorted(group.array_keys()):
        path = f'{prefix}{key}'
        try:
            yield path, group[key], None
        except Exception as exc:                          # noqa: BLE001
            yield path, None, exc
    for key in sorted(group.group_keys()):
        yield from walk(group[key], prefix=f'{prefix}{key}/')


def main():
    print(f'codecs: {CODECS}')
    store = zarr.ZipStore(str(ZIP), mode='r')
    root = zarr.open(store, mode='r')

    print()
    print('=' * 92)
    print('ARRAYS')
    print('=' * 92)
    print(f'{"path":36s} {"shape":>20s} {"dtype":>9s} {"chunks":>15s} '
          f'{"raw":>9s} {"compressor":>22s}')
    arrays = {}
    for path, a, err in walk(root):
        if err is not None:
            print(f'{path:36s} !! unreadable: {err.__class__.__name__}: {err}')
            continue
        arrays[path] = a
        comp = getattr(a.compressor, 'codec_id', None) or str(a.compressor)
        print(f'{path:36s} {str(a.shape):>20s} {str(a.dtype):>9s} '
              f'{str(a.chunks):>15s} {human(a.nbytes):>9s} {str(comp):>22s}')

    print()
    print('=' * 92)
    print('EPISODES')
    print('=' * 92)
    ends = None
    for key in ('meta/episode_ends', 'episode_ends'):
        if key in arrays:
            ends = np.asarray(arrays[key][:]).astype(np.int64)
            break
    if ends is None:
        print('no episode_ends found; keys: ' + ', '.join(sorted(arrays)))
    else:
        lens = np.diff(np.concatenate([[0], ends]))
        print(f'episodes           {len(ends)}')
        print(f'total steps        {int(ends[-1])}')
        print(f'steps per episode  min={lens.min()}  median={int(np.median(lens))}  '
              f'mean={lens.mean():.1f}  max={lens.max()}')

    print()
    print('=' * 92)
    print('WHAT THE AUDIO-ONLY PLAN READS')
    print('=' * 92)
    need = 0
    for path, a in arrays.items():
        if path.startswith('meta/'):
            continue
        is_rgb = 'rgb' in path or 'camera' in path
        tag = 'SKIP (vision)' if is_rgb else 'READ'
        if not is_rgb:
            need += a.nbytes
        print(f'  {tag:14s} {path:34s} {str(a.shape):>18s}  raw {human(a.nbytes):>9s}')
    print(f'\n  uncompressed total of the arrays we read: {human(need)}')
    print('  the RGB array is never read, which is what keeps this in RAM budget')

    print()
    print('=' * 92)
    print('SPOT CHECK')
    print('=' * 92)
    for path in ('data/mic_0', 'data/action', 'data/robot0_eef_pos',
                 'data/robot0_eef_rot_axis_angle', 'data/robot0_gripper_width'):
        if path not in arrays:
            print(f'{path:36s} MISSING')
            continue
        a = arrays[path]
        try:
            s = np.asarray(a[:3])
            print(f'{path:36s} slice{str(s.shape):>14s} dtype={str(s.dtype):>8s} '
                  f'min={s.min():+.5f} max={s.max():+.5f}')
        except Exception:                                 # noqa: BLE001
            print(f'{path:36s} read failed')
            traceback.print_exc()

    # Sample rate is not stored; infer the step rate from mic width.
    if 'data/mic_0' in arrays:
        w = arrays['data/mic_0'].shape[1]
        print()
        print(f'mic_0 is {w} samples per step. If the audio is 16 kHz that is '
              f'{16000/w:.1f} Hz steps;')
        print(f'if 48 kHz, {48000/w:.1f} Hz steps. ManiWAV resamples to 16 kHz '
              f'for the encoder.')
        if ends is not None:
            print(f'Total audio: {int(ends[-1])*w/16000:.1f} s at 16 kHz '
                  f'({int(ends[-1])*w/16000/60:.1f} min) across {len(ends)} episodes '
                  f'-> {int(ends[-1])*w/16000/len(ends):.1f} s per episode.')

    store.close()


if __name__ == '__main__':
    main()
