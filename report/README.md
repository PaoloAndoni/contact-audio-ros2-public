# Final report

The current report is [main.tex](main.tex), with all figures and references in
this folder. The compiled [PDF](../output/pdf/main.pdf) has 12 pages.

## Build

Use a TeX installation with the TU Darmstadt `tudapub` class (TUDa-CI), TikZ,
PGFPlots, and the packages listed in `main.tex`.

From the repository root, with Tectonic installed:

```bash
mkdir -p output/pdf
tectonic -X compile report/main.tex --outdir output/pdf
```

Alternatively, upload this folder to a TeX editor with TUDa-CI available,
select LuaLaTeX, and build with BibTeX. The class, TeX tools, and fonts are
external dependencies. They are not copied into this repository.

The figure inputs are already saved, so rebuilding the report does not need
the raw recordings or model training. `figures/home_embedding.tex` supplies
the corrected titles and probability labels for the saved scatter plot.

The report describes the limits of the microphone tests and the ManiWAV
experiment. Read it together with the [evaluation guide](../lab_audio_eval_2026-09-18/README.md).
