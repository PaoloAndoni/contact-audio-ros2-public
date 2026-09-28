import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import audio_io, quality

OUT_ROOT = pathlib.Path.home() / 'analysis' / 'pgf_export'
HEAT = OUT_ROOT / 'heat'
DAT = OUT_ROOT / 'dat'
TEX = OUT_ROOT / 'tex'

BAGS = pathlib.Path.home() / 'home_bags'
OUT = pathlib.Path.home() / 'analysis' / 'out_home'

disp = dict(np.load(OUT / 'display_spectrograms.npz'))

CLIPS = [
    'table_cleaning_noiseoff_01', 'table_cleaning_noiseon_02',
    'put_sizes_noiseoff_01', 'put_sizes_noiseon_01',
    'red_green_noiseoff_01', 'red_green_noiseon_01',
]

VMIN, VMAX = -68.0, -30.0
N_BUCKETS = 1400

def envelope(raw, n_buckets):
    n = len(raw)
    edges = np.linspace(0, n, n_buckets + 1).astype(int)
    pts = []
    for i in range(n_buckets):
        a, b = edges[i], edges[i+1]
        if b <= a:
            continue
        seg = raw[a:b]
        pts.append((a, seg.min()))
        pts.append((b-1, seg.max()))
    return np.array(pts)

DIGIT_MAP = str.maketrans('0123456789', 'zabcdefghi')

def safe(name):
    return name.replace('_', '').translate(DIGIT_MAP)

summary = {}
for name in CLIPS:
    L = audio_io.load_bag(BAGS / name)
    mono = audio_io.to_mono_float(L)
    raw = L['data'][:, 0].astype(np.float64)
    sr = L['sample_rate']
    q = quality.analyse(L, mono)
    v, _ = quality.verdict(q)

    t = np.arange(len(raw)) / sr
    env = envelope(raw, N_BUCKETS)
    ts = env[:, 0] / sr
    vs = env[:, 1]
    np.savetxt(DAT / f'{name}_wave.dat', np.column_stack([ts, vs]),
               header='t v', comments='', fmt='%.5f %.1f')

    fb = disp[name]
    db = fb.T * (10.0 / np.log(10.0))
    db_clipped = np.clip(db, VMIN, VMAX)
    norm = (db_clipped - VMIN) / (VMAX - VMIN)
    cmap = plt.get_cmap('magma')
    rgba = cmap(norm)
    plt.imsave(HEAT / f'{name}_heat.png', rgba, origin='lower')

    duration = len(raw) / sr
    sd = float(raw.std())
    lim = max(np.abs(raw).max() * 1.15, 8 * sd)
    events_s = [(a * 0.01, b * 0.01) for a, b in q['events']]
    ev_rows = ['t y']
    for a, b in events_s:
        ev_rows.append(f'{a:.3f} 124')
        ev_rows.append(f'{b:.3f} 124')
        ev_rows.append('nan nan')
    if not events_s:
        ev_rows.append('nan nan')
    (DAT / f'{name}_events.dat').write_text('\n'.join(ev_rows) + '\n')
    ev_str = ','.join(f'{a:.3f}/{b:.3f}' for a, b in events_s)

    summary[name] = dict(
        verdict=v, peak_counts=int(np.abs(raw).max()), peak_dbfs=q['peak_dbfs'],
        clip_sigma=sd, n_events=q['n_events'], duration=duration, ylim=lim,
        n_mel=fb.shape[1], n_frames=fb.shape[0],
    )

    tex_lines = []
    tex_lines.append(fr'\def\clip{safe(name)}Verdict{{{v}}}')
    tex_lines.append(fr'\def\clip{safe(name)}Peak{{{int(np.abs(raw).max())}}}')
    tex_lines.append(fr'\def\clip{safe(name)}PeakDbfs{{{q["peak_dbfs"]:.1f}}}')
    tex_lines.append(fr'\def\clip{safe(name)}Std{{{sd:.1f}}}')
    tex_lines.append(fr'\def\clip{safe(name)}SixSigma{{{6*sd:.1f}}}')
    tex_lines.append(fr'\def\clip{safe(name)}NEvents{{{q["n_events"]}}}')
    tex_lines.append(fr'\def\clip{safe(name)}Duration{{{duration:.2f}}}')
    tex_lines.append(fr'\def\clip{safe(name)}Ylim{{{lim:.0f}}}')
    tex_lines.append(fr'\def\clip{safe(name)}NMel{{{fb.shape[1]}}}')
    tex_lines.append(fr'\def\clip{safe(name)}NFrames{{{fb.shape[0]}}}')
    tex_lines.append(fr'\def\clip{safe(name)}PctShown{{{lim/32768*100:.1f}}}')
    tex_lines.append(fr'\def\clip{safe(name)}Events{{{ev_str}}}')
    (TEX / f'{name}.tex').write_text('\n'.join(tex_lines) + '\n')
    print(name, v, f'{q["peak_dbfs"]:.1f}dBFS', 'events=', q['n_events'])

(OUT_ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))
print('done')
