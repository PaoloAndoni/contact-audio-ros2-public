#!/usr/bin/env python3
"""Level gate: is the microphone actually delivering signal, before we record 18 takes?

This is the check whose absence cost the 18 Sept session. That session recorded
4 minutes of audio at a peak of 28 counts out of 32768 and nobody knew until it
was analysed at home. The rule here is blunt: tap the mic hard against the
object, and if the peak does not clear the threshold, do not record.

Reference points, all measured:
    ManiWAV contact mic   ~16000 counts peak   (a working contact-audio rig)
    18 Sept lab session      28 counts peak    (the failure)
    threshold used here    3277 counts         (-20 dBFS)
"""
import sys
import time

import numpy as np

TARGET_RATE = 48000
SECONDS = 5.0
GATE_DBFS = -20.0
GATE_COUNTS = 32768 * 10 ** (GATE_DBFS / 20.0)


def db(x):
    return 20.0 * np.log10(max(float(x), 1e-12))


def main():
    try:
        import sounddevice as sd
    except Exception as exc:                                  # noqa: BLE001
        print(f'FAIL: sounddevice not importable: {exc}')
        return 2

    print('=' * 72)
    print('AUDIO DEVICES VISIBLE TO WSL')
    print('=' * 72)
    try:
        devices = sd.query_devices()
        print(devices)
    except Exception as exc:                                  # noqa: BLE001
        print(f'FAIL: could not query devices: {exc}')
        return 2

    inputs = [(i, d) for i, d in enumerate(devices)
              if d.get('max_input_channels', 0) > 0]
    if not inputs:
        print('\nFAIL: no input devices at all. WSLg is not passing a mic through.')
        return 2

    print('\ninput-capable devices:')
    for i, d in inputs:
        print(f'  [{i}] {d["name"]}  ch={d["max_input_channels"]}  '
              f'default_sr={d["default_samplerate"]:.0f}')

    # Prefer the default input; fall back to the first input-capable device.
    try:
        default_in = sd.default.device[0]
        if default_in is None or default_in < 0:
            default_in = inputs[0][0]
    except Exception:                                         # noqa: BLE001
        default_in = inputs[0][0]
    dev = devices[default_in]
    print(f'\nusing device [{default_in}] {dev["name"]}')

    # Try 48 kHz first (matches the lab bags); fall back to the device default.
    rate = TARGET_RATE
    try:
        sd.check_input_settings(device=default_in, samplerate=rate, channels=1)
    except Exception:                                         # noqa: BLE001
        rate = int(dev['default_samplerate'])
        print(f'  48000 Hz refused; falling back to {rate} Hz')
    print(f'  rate {rate} Hz, 1 channel, int16')

    print()
    print('=' * 72)
    print(f'RECORDING {SECONDS:.0f} SECONDS -- TAP THE MIC AGAINST THE OBJECT NOW')
    print('=' * 72)
    for n in (3, 2, 1):
        print(f'  {n}...', flush=True)
        time.sleep(1)
    print('  GO -- tap hard, several times', flush=True)

    try:
        rec = sd.rec(int(SECONDS * rate), samplerate=rate, channels=1,
                     dtype='int16', device=default_in)
        sd.wait()
    except Exception as exc:                                  # noqa: BLE001
        print(f'FAIL: capture raised: {exc}')
        return 2

    x = np.asarray(rec, dtype=np.int64).reshape(-1)
    if x.size == 0:
        print('FAIL: captured zero samples.')
        return 2

    peak = int(np.abs(x).max())
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    # Clipping matters as much as level: a sample at the rail has had its true
    # value destroyed, and no amount of later processing recovers it. The first
    # version of this gate only tested "loud enough", which passed a take that
    # was pinned to full scale.
    clipped = int((np.abs(x) >= 32767).sum())
    pct_clipped = 100.0 * clipped / x.size

    # Flat-top detection, independent of absolute level. Digital attenuation
    # applied AFTER a converter has already clipped will scale the flattened
    # peaks down below full scale -- the damage is still there, it just no
    # longer shows up as "at 32767". A real waveform almost never repeats its
    # own maximum for several samples in a row; a clipped one always does.
    mx = int(np.abs(x).max())
    at_max = np.abs(x) >= mx * 0.995
    runs, run = [], 0
    for v in at_max:
        if v:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    longest_flat = max(runs) if runs else 0
    n_flat_runs = sum(1 for r in runs if r >= 3)
    # Noise floor: median of the quietest 20% of 25 ms frames.
    n = int(rate * 0.025)
    frames = x[:len(x) // n * n].reshape(-1, n)
    fr = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    floor = float(np.median(fr[fr <= np.percentile(fr, 20)]))
    distinct = int(len(np.unique(x)))

    print()
    print('=' * 72)
    print('RESULT')
    print('=' * 72)
    print(f'  peak          {peak:6d} counts   ({db(peak/32768):+7.1f} dBFS)')
    print(f'  rms           {rms:9.1f} counts   ({db(rms/32768):+7.1f} dBFS)')
    print(f'  noise floor   {floor:9.1f} counts   ({db(floor/32768):+7.1f} dBFS)')
    print(f'  peak / floor  {peak/max(floor,1e-9):9.1f}x  '
          f'({db(peak/max(floor,1e-9)):+.1f} dB)')
    print(f'  clipped       {clipped:6d} samples ({pct_clipped:.3f}%)')
    print(f'  flat-topping  {n_flat_runs:6d} runs of >=3 samples at the peak '
          f'(longest {longest_flat})')
    print(f'  distinct codes{distinct:6d} of 65536')
    print()
    print('  target band: peak -12 to -6 dBFS (8200-16400 counts), 0% clipped')
    print('  for scale:   18 Sept lab session peaked at 28 counts;')
    print('               ManiWAV contact mic peaks around 16000.')
    print()

    # Too hot is a failure, not a pass. Clipped samples cannot be recovered.
    if pct_clipped > 0.01 or peak >= 32767:
        print(f'  FAIL (TOO HOT) -- {clipped} samples at full scale.')
        print('  Those peaks are flattened and unrecoverable. Reduce the level')
        print('  and re-run before recording anything.')
        return 3
    if n_flat_runs > 5:
        print(f'  FAIL (CLIPPED UPSTREAM) -- peak is {peak} counts, below full '
              f'scale, but {n_flat_runs} runs sit flat at that peak.')
        print('  That is a waveform already clipped in the microphone before')
        print('  any software attenuation. Turning the digital volume down')
        print('  cannot repair it -- the level must come down BEFORE the')
        print('  converter: move the mic further from the source, or loosen')
        print('  its contact with the object.')
        return 3
    if peak >= GATE_COUNTS:
        head = db(32768 / max(peak, 1))
        print(f'  PASS -- peak {peak} counts ({db(peak/32768):+.1f} dBFS), '
              f'{head:.1f} dB of headroom, nothing clipped.')
        if peak > 26000:
            print('  Note: close to the rail. A harder contact could clip;')
            print('  consider dropping the input volume one more notch.')
        print('  This microphone is usable. Go ahead and record.')
        return 0
    if peak >= 512:
        print(f'  MARGINAL -- peak {peak} is below {GATE_COUNTS:.0f} '
              f'({GATE_DBFS:.0f} dBFS) but well above the lab failure.')
        print('  Raise the input gain or move the mic closer, then re-run.')
        return 1
    print(f'  FAIL -- peak {peak} counts. This is the 18 Sept failure again.')
    print('  Do NOT record. Check: input gain, selected device, mute switch,')
    print('  and that the mic is physically contacting the object.')
    return 2


if __name__ == '__main__':
    sys.exit(main())
