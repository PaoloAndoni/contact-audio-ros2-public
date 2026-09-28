# Public copy: contents and checks

Prepared on September 28, 2026 from `PaoloAndoni/contact-audio-ros2`, commit
`81cd1271fbeff4c7111308352d30c882936c8ddd`.

## What was missing and added

- A main README with setup and launch steps.
- The final 12-page report, its source, figures, and references.
- The later home-audio and ManiWAV evaluation scripts and saved results.
- A data guide explaining the inputs needed to repeat each experiment.
- A license file matching the MIT license already declared in the ROS packages.
- A combined Python dependency list and separate ManiWAV dependencies.

The original ROS source matched the saved local lab bundle. It is preserved in
this copy. The later analysis scripts now use paths within this repository, with
environment variables for other locations. A few comments were corrected to
match the final report. The experiments and saved scores were not changed.

## Checks run for this copy

- 57 existing tests passed: pose maths, action chunking, and observation assembly.
- All Python files were checked for syntax.
- ROS package manifests and saved JSON files were read successfully.
- The AST, CNN, and mel-statistics model bundles loaded successfully. The
  bundled CNN weights matched their saved checksum.
- Local links in the README files were checked.
- Common credential patterns were checked before publication; none were found.

Live ROS, microphone capture, and the full audio-model tests were not rerun
on this Windows machine. The analysis models were not retrained.

## Inputs kept outside the repository

Raw ROS bags, original training WAVs, large derived arrays, and downloaded
ManiWAV data are not included. The saved results can be read immediately, but
exactly repeating the home experiment requires its original recordings.
See the [evaluation guide](lab_audio_eval_2026-09-18/README.md).

The older `scripts/lab_record.sh` requests MCAP storage. Install the MCAP ROS
storage plugin before using that script, or select SQLite in the script on a
Humble machine. The final home evaluation uses SQLite bags. Older demo and
verification scripts may also need their original machine paths changed.

The compiled PDF is included for reading and download. A separate public
blogpost page is still needed for the course's report-URL requirement.
