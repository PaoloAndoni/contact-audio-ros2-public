# Audio for Contact-Rich Manipulation

Project lab by **Paolo Andoni**, supervised by **Rickmer Krohn**.

This project adds audio features to a ROS 2 data pipeline for robot manipulation.
It records sound, makes mel spectrograms, extracts audio features, and pairs
observations with future motion targets. It does not control a robot arm.

**[Read the final report (PDF)](output/pdf/main.pdf)** ·
**[Report source](report/main.tex)** ·
**[Evaluation guide](lab_audio_eval_2026-09-18/README.md)**

## What is included

| Folder | Contents |
| --- | --- |
| `src/contact_audio_interfaces` | ROS 2 message and service definitions |
| `src/contact_audio_perception` | Audio nodes, encoders, model bundles, and tests |
| `scripts` | Training, recording, alignment, and checking tools |
| `lab_audio_eval_2026-09-18` | Later experiments and the saved report results |
| `report` | Final LaTeX source, figures, and 18 references |
| `output/pdf/main.pdf` | Final 12-page report |
| `docs` | Earlier pipeline results; these are not the final report |

The recordings in the final report use an **air microphone** and hand-made
contact sounds. They do not test a mounted contact microphone or a learned
robot policy. The ManiWAV study is a separate, small action-prediction test.

## Set up ROS 2

Use Ubuntu 24.04 with ROS 2 Jazzy. Install ROS 2 first using its
[installation guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
The lab recordings used Humble with SQLite bags; the full pipeline has not been
retested on Humble for this public copy.

Clone this repository as a workspace, then build from its root:

```bash
git clone https://github.com/PaoloAndoni/contact-audio-ros2-public.git
cd contact-audio-ros2-public
source /opt/ros/jazzy/setup.bash
sudo apt update
sudo apt install python3-colcon-common-extensions python3-rosdep python3-venv libportaudio2
# Run "sudo rosdep init" once if rosdep has not been set up on this machine.
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Keep Python packages in a local environment that can also see ROS packages.
python3 -m venv --system-site-packages .venv
touch .venv/COLCON_IGNORE
source .venv/bin/activate
python -m pip install -r requirements.txt
colcon build --symlink-install
source install/setup.bash
```

In each new terminal, source ROS, activate `.venv`, and source `install/setup.bash`.
The AST model is downloaded from Hugging Face on first use. Its model ID and
revision are stored in the supplied model bundle.

## Run the audio pipeline

With a microphone:

```bash
ros2 launch contact_audio_perception two_node.launch.py source:=mic
```

With your own 16-bit PCM WAV file:

```bash
ros2 launch contact_audio_perception two_node.launch.py \
  source:=wav wav_path:=/absolute/path/to/recording.wav
```

The feature node publishes at 15 Hz. Classification can be slower on a CPU.
The bundled classifiers use the earlier `no_contact`, `slide_table`, and
`tap_table` classes. The final report's three sound categories are evaluated
separately by the scripts in the evaluation folder.

## Check the code

After building and sourcing the workspace:

```bash
python -m pytest src/contact_audio_perception/test/test_pose_math.py \
  src/contact_audio_perception/test/test_action.py \
  src/contact_audio_perception/test/test_observation.py \
  src/contact_audio_perception/test/test_spectrogram.py
```

See [publication checks](PUBLICATION.md) for what was checked for this copy.

## Data and limits

Raw recordings, original training WAVs, large derived arrays, and downloaded
ManiWAV data are not included. The supplied model bundles and saved result files
are included. See the [data guide](lab_audio_eval_2026-09-18/README.md) for the
required inputs and steps to repeat the experiments.

The eight-split ManiWAV scores use full-dataset scaling and the best validation
epoch. They are early validation results, not an independent test score or
proof that audio improves robot control.

Some older scripts under `scripts/` use paths from the original machine. Read
their options and paths before using them. The main launch commands above and
the later evaluation scripts use this repository's layout.

## Source and license

This is a public copy of `PaoloAndoni/contact-audio-ros2`, with the final report
and later analysis added. The original repository remains unchanged.
The code uses the MIT license already declared in its ROS package files.
See [LICENSE](LICENSE) and [third-party notes](THIRD_PARTY.md).
