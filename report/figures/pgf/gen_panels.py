import json, pathlib

ROOT = pathlib.Path(__file__).parent
summary = json.loads((ROOT / 'summary.json').read_text())

TASK_COLOR = {
    'table_cleaning': 'TCGreen',
    'put_sizes': 'TCBlue',
    'red_green': 'TCOrange',
}

TASK_TITLE = {
    'table_cleaning': 'Cleaning: sound example',
    'put_sizes': 'Peg insertion: sound example',
    'red_green': 'Peg in hole: sound example',
}

TASKS = [
    ('table_cleaning', 'table_cleaning_noiseoff_01', 'table_cleaning_noiseon_02'),
    ('put_sizes', 'put_sizes_noiseoff_01', 'put_sizes_noiseon_01'),
    ('red_green', 'red_green_noiseoff_01', 'red_green_noiseon_01'),
]

# All lengths in cm (tikzpicture uses x=1cm,y=1cm).
WAVE_H = 2.9
SPEC_H = 3.7
COL_W = 6.55
GAP = 1.55
CB_W = 1.0
CB_GAP = 0.15
VMIN, VMAX = -68.0, -30.0

X_OFF = 0.0
X_ON = COL_W + GAP
X_CB = 2 * COL_W + GAP + CB_GAP
TITLE_Y = 1.05


def clip_label(name):
    return name.replace('_', '\\_')


def panel(name, x, color, show_ylabel, show_colorbar=False):
    m = summary[name]
    six_sigma = 6 * m['clip_sigma']
    pct = m['ylim'] / 32768 * 100
    ylabel_amp = r'amplitude (counts)' if show_ylabel else ''
    ylabel_mel = r'mel bin' if show_ylabel else ''
    spec_y = -(WAVE_H + 0.2)
    out = []
    out.append(f"""  \\begin{{axis}}[
    at={{({x:.2f}cm,0cm)}}, anchor=north west,
    width={COL_W:.2f}cm, height={WAVE_H:.2f}cm,
    xmin=0, xmax={m['duration']:.2f}, ymin=-{m['ylim']:.0f}, ymax={m['ylim']:.0f},
    title={{\\texttt{{\\scriptsize {clip_label(name)}}}\\\\[1pt]
      \\scriptsize {m['verdict']}\\ \\textperiodcentered\\ peak {m['peak_counts']}~cts ({m['peak_dbfs']:.1f}~dBFS) \\textperiodcentered\\ clip $\\sigma = ${m['clip_sigma']:.1f}~cts}},
    title style={{align=left, font=\\scriptsize, at={{(0,1.04)}}, anchor=south west}},
    ylabel={{{ylabel_amp}}}, xlabel={{}}, xticklabels=\\empty,
    tick label style={{font=\\scriptsize}}, label style={{font=\\scriptsize}},
    axis line style={{black!55}}, tick style={{black!55}},
    no markers,
  ]
    \\addplot[fill=black!8, draw=none] coordinates
      {{(0,-{six_sigma:.1f}) ({m['duration']:.2f},-{six_sigma:.1f}) ({m['duration']:.2f},{six_sigma:.1f}) (0,{six_sigma:.1f})}} -- cycle;
    \\addplot[{color}, line width=0.2pt] table {{figures/pgf/dat/{name}_wave.dat}};
    \\node[anchor=north east, font=\\tiny, text=black!45] at (rel axis cs:0.99,0.98)
      {{$\\pm6\\sigma$ band ({pct:.0f}\\% of full scale shown)}};
  \\end{{axis}}

  \\begin{{axis}}[
    at={{({x:.2f}cm,{spec_y:.2f}cm)}}, anchor=north west,
    width={COL_W:.2f}cm, height={SPEC_H:.2f}cm,
    xmin=0, xmax={m['duration']:.2f}, ymin=0, ymax={m['n_mel']},
    xlabel={{time (s)}}, ylabel={{{ylabel_mel}}},
    tick label style={{font=\\scriptsize}}, label style={{font=\\scriptsize}},
    axis line style={{black!55}}, tick style={{black!55}},
    enlargelimits=false, no markers,
    colormap name=magma, point meta min={VMIN:.0f}, point meta max={VMAX:.0f}{
      ', colorbar, colorbar style={width=0.28cm, tick label style={font=\\scriptsize}, ylabel={dB}, ylabel style={font=\\scriptsize, yshift=-2mm}, ytick={-70,-60,-50,-40,-30}}'
      if show_colorbar else ''},
  ]
    \\addplot graphics [xmin=0,xmax={m['duration']:.2f},ymin=0,ymax={m['n_mel']}]
      {{figures/pgf/heat/{name}_heat.png}};
    \\addplot[{color}, line width=1.0pt, unbounded coords=jump] table {{figures/pgf/dat/{name}_events.dat}};
    \\node[anchor=north west, font=\\tiny, text=white, fill=black, fill opacity=0.55,
      text opacity=1, inner sep=1.5pt] at (rel axis cs:0.01,0.98)
      {{{m['n_events']} detected event(s)}};
  \\end{{axis}}""")
    return '\n'.join(out)


for task, off_clip, on_clip in TASKS:
    color = TASK_COLOR[task]
    title = TASK_TITLE[task]
    parts = []
    parts.append(r'\begin{tikzpicture}[x=1cm,y=1cm]')
    parts.append(f"  \\node[anchor=south west, font=\\bfseries] at ({X_OFF:.2f},{TITLE_Y:.2f}) {{{title} --- noise off vs.\\ noise on}};")
    parts.append(panel(off_clip, X_OFF, color, True))
    parts.append(panel(on_clip, X_ON, color, False, show_colorbar=True))
    parts.append(r'\end{tikzpicture}')
    (ROOT / 'tex' / f'{task}_panel.tex').write_text('\n\n'.join(parts) + '\n')
    print('wrote', task)
