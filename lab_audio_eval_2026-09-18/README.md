# Experiments used in the report

This folder adds the experiments that were missing from the original ROS repo.
The folder name is the date analysis started; the home recordings were made on
September 23, 2026.

## Saved results

| File | Use |
| --- | --- |
| `clip_quality_home.csv` | Quality and signal-level checks for the home recordings |
| `separability_home.json` | Held-out clip classification and shuffled-label checks |
| `maniwav_multisplit.json` | Eight episode splits used in the final report |
| `maniwav_wipe_results.json` | Earlier single-split experiment |
| `clip_quality.csv`, `separability.json` | Earlier lab session; not the home results |

## Inputs not included

- Home ROS bags: 18 labelled 15-second clips, plus two noise-only recordings.
- Earlier WAV files used to fit the bundled contact classifiers.
- The ManiWAV `wipe` archive and the 78 derived episode arrays used here.
- Cached AST weights and per-window feature arrays. The code can generate these.

The home recordings cannot be rebuilt from the summary files. They must be
provided separately to repeat that part of the experiment exactly. These scripts
also accept new recordings with the same message type and naming scheme.

## Home recordings

Build and source the ROS workspace as described in the main README. Each bag
must contain `/audio/raw_chunk` as `contact_audio_interfaces/msg/AudioChunk`
in SQLite (`.db3`) format. Put the bag folders in `data/home_bags`, or set
`BAGS_DIR` to their location.

Names have the form `<sound>_<noise>_<take>`, for example
`put_sizes_noiseoff_01`. The sound codes mean:

| Code | Recorded sound |
| --- | --- |
| `table_cleaning` | Felt moved over paper |
| `put_sizes` | A coin, razor, or cup placed on a surface |
| `red_green` | A cup placed in a plate |

These are substitutes for the planned robot tasks, not actual peg insertion
trials. The two background recordings are `noise_only_off` and `noise_only_on`.

From the repository root:

```bash
export BAGS_DIR="$PWD/data/home_bags"
export OUT_DIR="$PWD/output/evaluation/home"
export FIG_DIR="$PWD/output/evaluation/home/figures"
python lab_audio_eval_2026-09-18/scripts/inspect_bags.py
python lab_audio_eval_2026-09-18/scripts/run_quality.py
python lab_audio_eval_2026-09-18/scripts/diagnose.py
python lab_audio_eval_2026-09-18/scripts/features.py
python lab_audio_eval_2026-09-18/scripts/separability.py
python lab_audio_eval_2026-09-18/scripts/viz.py
```

`features.py` makes one-second windows every 0.5 seconds. `separability.py`
holds out one whole clip, fits class means on the other clips, predicts each
window by cosine similarity, and uses a majority vote for the clip. It does
not use the existing classifier's class labels for this evaluation.

## ManiWAV action prediction

Get the `wipe` dataset from the
[ManiWAV project](https://maniwav.cs.columbia.edu/). The original download used
[this archive](https://real.stanford.edu/maniwav/data/wipe/replay_buffer.zarr.zip).
Place it at `data/maniwav/wipe_replay_buffer.zarr.zip`, or set `MANIWAV_ZIP`.
The original dataset and pretrained model keep their own terms of use.

```bash
python -m pip install -r lab_audio_eval_2026-09-18/requirements-maniwav.txt
export OUT_DIR="$PWD/output/evaluation/maniwav"
python lab_audio_eval_2026-09-18/scripts/extract_embed.py --episodes 78
python lab_audio_eval_2026-09-18/scripts/train_multisplit.py --splits 8 --epochs 40
```

The defaults use `mic_0`, an assumed 48 kHz audio rate and 60 Hz pose rate,
one-second windows, and an embedding every four pose steps. The model sees
30 audio windows and/or the current pose plus gripper state. It predicts
16 consecutive pose changes. These targets differ from ManiWAV's policy targets.

The code keeps the settings behind the saved report results: full-dataset
normalisation, a full-dataset mean baseline, and the lowest validation loss
over 40 epochs. There is no separate test set. This should be changed in a new
study, but changing it here would no longer reproduce the reported experiment.
The older `train_head.py` uses training-only scaling for its single split.

Path settings are in `scripts/paths.py`. `DERIVED_DIR`, `MODELS_DIR`, `OUT_DIR`,
and `FIG_DIR` can also be set. Outputs are kept separate from the saved results.
No models were retrained when this public copy was prepared.
