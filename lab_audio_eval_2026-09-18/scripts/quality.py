"""Per-clip audio quality and contact-event SNR.

Every number here is measured from the decoded samples. Nothing is assumed
from the recording settings.
"""
import numpy as np

FRAME_MS = 25.0
HOP_MS = 10.0
# Full-scale for int16 audio expressed in the [-1, 1) float domain.
EPS = 1e-12


def db(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), EPS))


def frame_rms(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    n = int(round(sr * frame_ms / 1000.0))
    h = int(round(sr * hop_ms / 1000.0))
    if len(x) < n:
        return np.zeros(0), np.zeros(0)
    starts = np.arange(0, len(x) - n + 1, h)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    t = starts / sr
    return rms, t


def analyse(loaded, mono):
    """Quality + event SNR for one clip.

    Noise floor is the median of the quietest 20% of frames; an event frame is
    one that exceeds the floor by EVENT_MARGIN_DB. SNR is the ratio of mean
    event power to noise-floor power, which is the quantity that matters for
    whether a contact is separable from the room.
    """
    sr = loaded['sample_rate']
    peak = float(np.abs(mono).max()) if len(mono) else 0.0
    rms = float(np.sqrt((mono.astype(np.float64) ** 2).mean())) if len(mono) else 0.0

    # Clipping measured on the raw int16, where full scale is unambiguous.
    raw = loaded['data']
    clipped = int((np.abs(raw) >= 32767).sum())
    pct_clipped = 100.0 * clipped / max(raw.size, 1)

    # Effective resolution actually used: how many int16 codes the signal spans.
    peak_counts = float(np.abs(raw).max())
    used_bits = float(np.log2(peak_counts + 1)) if peak_counts > 0 else 0.0

    fr, ft = frame_rms(mono, sr)
    if len(fr) == 0:
        return {'peak': peak, 'rms': rms, 'pct_clipped': pct_clipped,
                'n_events': 0, 'snr_db': float('nan')}

    frdb = db(fr)
    quiet_cut = np.percentile(frdb, 20)
    floor_db = float(np.median(frdb[frdb <= quiet_cut]))
    floor_lin = 10 ** (floor_db / 20.0)

    EVENT_MARGIN_DB = 6.0
    thresh_db = floor_db + EVENT_MARGIN_DB
    active = frdb > thresh_db

    # Group contiguous active frames into events; require >= 3 frames (~45 ms)
    # so single-frame noise ticks are not counted as contacts.
    events = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if j - i >= 3:
                events.append((i, j))
            i = j
        else:
            i += 1

    if events:
        ev_mask = np.zeros(len(fr), dtype=bool)
        for a, b in events:
            ev_mask[a:b] = True
        ev_power = float((fr[ev_mask] ** 2).mean())
        ev_rms = float(np.sqrt(ev_power))
        ev_peak = float(fr[ev_mask].max())
        snr_db = float(10 * np.log10(max(ev_power, EPS) / max(floor_lin ** 2, EPS)))
        peak_snr_db = float(20 * np.log10(max(ev_peak, EPS) / max(floor_lin, EPS)))
        ev_time = float(sum(b - a for a, b in events) * HOP_MS / 1000.0)
    else:
        ev_rms = float('nan')
        snr_db = float('nan')
        peak_snr_db = float('nan')
        ev_time = 0.0

    return {
        'sample_rate': sr,
        'channels': loaded['channels'],
        'duration_s': loaded['duration_s'],
        'n_chunks': loaded['n_chunks'],
        'gap_count': len(loaded['gaps']),
        'gap_seconds': loaded['gap_seconds'],
        'overlap_frames': loaded['overlap_frames'],
        'peak': peak,
        'peak_dbfs': float(db(peak)),
        'peak_counts': peak_counts,
        'used_bits': used_bits,
        'rms': rms,
        'rms_dbfs': float(db(rms)),
        'pct_clipped': pct_clipped,
        'floor_dbfs': floor_db,
        'n_events': len(events),
        'event_rms_dbfs': float(db(ev_rms)) if ev_rms == ev_rms else float('nan'),
        'event_time_s': ev_time,
        'snr_db': snr_db,
        'peak_snr_db': peak_snr_db,
        'frame_rms': fr,
        'frame_t': ft,
        'events': events,
        'floor_lin': floor_lin,
    }


def verdict(q):
    """Plain usable / marginal / unusable call, with the reason that drove it."""
    reasons = []
    bad = False
    marginal = False

    if q['peak_counts'] < 64:
        reasons.append(
            f'peak is only {q["peak_counts"]:.0f}/32768 counts '
            f'({q["peak_dbfs"]:.1f} dBFS) -- ~{q["used_bits"]:.1f} of 16 bits used')
        bad = True
    elif q['peak_counts'] < 512:
        reasons.append(
            f'peak {q["peak_counts"]:.0f}/32768 counts ({q["peak_dbfs"]:.1f} dBFS), '
            f'~{q["used_bits"]:.1f} bits used')
        marginal = True

    if q['pct_clipped'] > 0.1:
        reasons.append(f'{q["pct_clipped"]:.2f}% of samples clipped')
        bad = True

    if q['gap_seconds'] > 0.05:
        reasons.append(f'{q["gap_count"]} dropouts totalling {q["gap_seconds"]*1000:.0f} ms')
        marginal = True

    s = q['snr_db']
    if s != s:
        reasons.append('no contact events detected above the noise floor')
        bad = True
    elif s < 6:
        reasons.append(f'event-to-floor SNR only {s:.1f} dB')
        bad = True
    elif s < 12:
        reasons.append(f'event-to-floor SNR {s:.1f} dB')
        marginal = True
    else:
        reasons.append(f'event-to-floor SNR {s:.1f} dB')

    if bad:
        return 'UNUSABLE', reasons
    if marginal:
        return 'MARGINAL', reasons
    return 'USABLE', reasons
