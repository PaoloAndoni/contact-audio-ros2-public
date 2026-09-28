#!/usr/bin/env bash
# One-command evidence for the spectrogram-first rebuild.
#
#   ./scripts/verify_sprint.sh              portable, offline, no microphone
#   ./scripts/verify_sprint.sh --full-eval  also recompute the 4-extractor CV
#   ./scripts/verify_sprint.sh --mic        strict native-microphone acceptance
#
# The default mode is what you run in front of someone: it proves the rebuilt
# pipeline reproduces the old one exactly, installs correctly, and runs live off
# a WAV file. --mic is the gate that decides whether this machine may record
# synchronised bags; on WSLg it is EXPECTED to fail, and that failure must not
# be dressed up as a pass.
# No `set -u`: ROS 2's setup.bash references unbound variables and aborts under it.
# `set -m` gives every background job its own process group, so the whole group
# can be signalled by PID. (setsid is not used here: it forks when the caller is
# already a group leader, which makes $! the wrong pid and a group kill land on
# the script itself.)
set -o pipefail
set -m

FULL_EVAL=0
MIC=0
for arg in "$@"; do
  case "$arg" in
    --full-eval) FULL_EVAL=1 ;;
    --mic) MIC=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS" || exit 1
RAW="${RAW_DIR:-$HOME/train_data}"
CHILDREN=()

# Only ever kill what this script started. A blanket pkill would take out a
# colleague's nodes on a shared machine.
# Signal the child's whole process group. Killing only the direct child is not
# enough: `ros2 run` leaves the node orphaned, and a surviving DAQ publishing
# onto the same topic as the next step's DAQ corrupts every timestamp downstream
# -- which is exactly how this script first reported a bogus -1407 ms age.
SELF_PGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')

# Signal a child's process group, but NEVER our own: if job control did not give
# some job its own group, a group kill would take out this script and whatever
# launched it. Fall back to signalling the single pid in that case.
# `set -m` guarantees a background job's PGID equals its PID, so the group can be
# signalled from the PID alone. Looking the PGID up with ps at cleanup time is
# what leaked nodes before: once the direct child had exited, ps returned
# nothing, the group kill was skipped, and `ros2 run`'s grandchild survived to
# publish into the next step.
kill_child() {
  local sig="$1" pid="$2"
  case "$pid" in ''|*[!0-9]*) return ;; esac
  [ "$pid" != "$SELF_PGID" ] && kill "-$sig" -- "-$pid" 2>/dev/null
  kill "-$sig" "$pid" 2>/dev/null
  return 0
}

cleanup() {
  local rc=$? pid          # preserve the script's exit status across the trap
  for pid in "${CHILDREN[@]}"; do kill_child TERM "$pid"; done
  sleep 1
  for pid in "${CHILDREN[@]}"; do
    kill_child KILL "$pid"
    case "$pid" in ''|*[!0-9]*) continue ;; esac
    wait "$pid" 2>/dev/null
  done
  CHILDREN=()
  return $rc
}
trap cleanup EXIT INT TERM

spawn() { "$@" & CHILDREN+=($!); }

PIPELINE_NODES='audio_daq_node|contact_ml_node|wav_player_node|audio_window_node|ast_encoder_node|contact_classifier_node'

# Killing the processes is not the same as the graph being empty. A node that
# outlives its step keeps publishing onto the next step's topics, and two DAQs
# on one topic make every measured timestamp meaningless -- so wait for the
# graph to drain before trusting anything downstream.
settle() {
  local waited=0
  while [ "$waited" -lt 15 ]; do
    if ! ros2 node list 2>/dev/null | grep -qE "$PIPELINE_NODES"; then return 0; fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "  WARNING: pipeline nodes still present after cleanup:"
  ros2 node list 2>/dev/null | grep -E "$PIPELINE_NODES" | sed 's/^/    /'
  return 1
}

FAILURES=()
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
record() {
  if [ "$1" -eq 0 ]; then echo "  -> PASS: $2"; else echo "  -> FAIL: $2"; FAILURES+=("$2"); fi
}

source /opt/ros/jazzy/setup.bash

step "1. Build"
colcon build 2>&1 | tail -3
record $? "build"
source install/setup.bash
SHARE=$(python3 -c "from ament_index_python.packages import get_package_share_directory as g; print(g('contact_audio_perception'))")
echo "  commit: $(git rev-parse --short HEAD)  share: $SHARE"

step "2. Tests"
colcon test --packages-select contact_audio_perception 2>&1 | tail -2
colcon test-result --all 2>&1 | tail -2
colcon test-result --all >/dev/null 2>&1
record $? "unit tests"

step "3. Installed model bundles"
python3 - "$SHARE" <<'PY'
import sys
from pathlib import Path
from contact_audio_perception.contact_model import ContactModel, bundle_paths
bundles = bundle_paths(Path(sys.argv[1]) / 'models')
if not bundles:
    print('no bundles installed'); raise SystemExit(1)
for path in bundles:
    model = ContactModel.load(path)
    extra = ''
    if model.weights_source == 'bundled':
        tensors = model.load_weights()
        extra = f', {len(tensors)} tensors, checksum verified'
    print(f'  {path.name:<26} {model.encoder_id}  fold={model.fold_id}{extra}')
PY
record $? "bundle integrity"

step "4. AST parity gates 1-3 (bit-exact against the legacy pipeline)"
python3 scripts/verify_block2.py "$RAW" --model "$SHARE/models/contact_model_ast" 2>&1 \
  | grep -E "^GATE|^RESULT|windows tested"
record ${PIPESTATUS[0]} "AST parity gates"

step "5. Gate 4: two-node vs four-node over ROS (deterministic WAV)"
LONG=/tmp/verify_sprint_long.wav
python3 - "$RAW" "$LONG" <<'PY'
import sys, wave
src = [f'{sys.argv[1]}/tap_table/tap_table_{i:03d}.wav' for i in (1, 2, 3, 4)]
with wave.open(src[0], 'rb') as f:
    params = f.getparams()
frames = b''.join(wave.open(p, 'rb').readframes(wave.open(p, 'rb').getnframes()) for p in src)
with wave.open(sys.argv[2], 'wb') as out:
    out.setparams(params); out.writeframes(frames)
PY
spawn ros2 run contact_audio_perception contact_ml_node --ros-args \
  -p model_path:="$SHARE/models/contact_model_ast" >/tmp/vs_ml.log 2>&1
sleep 25
spawn python3 scripts/verify_block2_ros.py --record /tmp/vs_new.jsonl 45
sleep 2
ros2 run contact_audio_perception audio_daq_node --ros-args \
  -p source:=wav -p wav_path:="$LONG" -p publish_rate_hz:=0.5 \
  -p wav_realtime:=false -p shutdown_when_done:=true >/tmp/vs_daq.log 2>&1
wait "${CHILDREN[-1]}" 2>/dev/null
cleanup; settle
spawn ros2 run contact_audio_perception audio_window_node --ros-args \
  -p window_sec:=1.0 -p hop_sec:=0.5 -p mono:=true >/tmp/vs_win.log 2>&1
spawn ros2 run contact_audio_perception ast_encoder_node >/tmp/vs_ast.log 2>&1
spawn ros2 run contact_audio_perception contact_classifier_node --ros-args \
  -p model_path:="$SHARE/models/contact_centroids.npz" >/tmp/vs_clf.log 2>&1
sleep 25
spawn timeout 100 ros2 run contact_audio_perception wav_player_node --ros-args \
  -p wav_path:="$LONG" -p loop:=true >/tmp/vs_wav.log 2>&1
sleep 2
python3 scripts/verify_block2_ros.py --record /tmp/vs_legacy.jsonl 95 >/dev/null
cleanup; settle
python3 scripts/verify_block2_ros.py /tmp/vs_new.jsonl /tmp/vs_legacy.jsonl
record $? "Gate 4 ROS equivalence"

step "6. Timing control (WAV source: derivation and harness are sound)"
spawn ros2 run contact_audio_perception contact_ml_node --ros-args \
  -p model_path:="$SHARE/models/contact_model_cnn" >/tmp/vs_mlc.log 2>&1
sleep 8
spawn ros2 run contact_audio_perception audio_daq_node --ros-args \
  -p source:=wav -p wav_path:="$RAW/tap_table/tap_table_001.wav" -p loop:=true \
  -p publish_rate_hz:=15.0 >/tmp/vs_daqc.log 2>&1
sleep 3
python3 scripts/characterize_timing.py 40 --source wav --require-predictions
record $? "WAV timing control"
cleanup; settle

step "7. Live two-node pipeline on WAV (DAQ + CNN)"
spawn ros2 run contact_audio_perception contact_ml_node --ros-args \
  -p model_path:="$SHARE/models/contact_model_cnn" >/tmp/vs_mll.log 2>&1
sleep 8
spawn ros2 run contact_audio_perception audio_daq_node --ros-args \
  -p source:=wav -p wav_path:="$RAW/tap_table/tap_table_001.wav" -p loop:=true \
  -p publish_rate_hz:=15.0 >/tmp/vs_daql.log 2>&1
sleep 10
timeout 10 ros2 topic hz /audio/spectrogram --window 60 >/tmp/vs_hz_spec.txt 2>&1
timeout 10 ros2 topic hz /audio/contact_prediction --window 60 >/tmp/vs_hz_pred.txt 2>&1
cleanup; settle
python3 - <<'PY'
import re, sys
def rate(path):
    text = open(path).read()
    values = [float(m) for m in re.findall(r'average rate:\s*([\d.]+)', text)]
    return max(values) if values else 0.0
spec, pred = rate('/tmp/vs_hz_spec.txt'), rate('/tmp/vs_hz_pred.txt')
print(f'  spectrogram {spec:.2f} Hz, predictions {pred:.2f} Hz (expect ~15 Hz each)')
sys.exit(0 if spec >= 13.0 and pred >= 13.0 else 1)
PY
record $? "live two-node pipeline at 15 Hz"

step "8. Encoder comparison"
if [ "$FULL_EVAL" -eq 1 ]; then
  python3 scripts/evaluate_encoders.py "$RAW" --out eval_runs/verify_sprint 2>&1 \
    | tail -8
  record $? "four-extractor cross-validation (recomputed)"
else
  python3 - <<'PY'
import json, pathlib
p = pathlib.Path('eval_runs/full/results.json')
if not p.is_file():
    print('  no stored results; run with --full-eval'); raise SystemExit(0)
d = json.loads(p.read_text())
roles = {'ast': 'pretrained neural', 'ast_random': 'pretraining control',
         'cnn': 'learned neural', 'melstats': 'non-neural baseline'}
print(f'  {"extractor":<12}{"role":<22}{"clip":>8}{"window":>9}{"ms/win":>9}{"params":>12}')
for k, r in d['results'].items():
    print(f'  {k:<12}{roles[k]:<22}{r["per_clip"]:>8.3f}{r["per_window"]:>9.3f}'
          f'{r["latency_ms"]:>9.1f}{r["parameters"]:>12,}')
print(f'  dataset {d["dataset_digest"]}, {d["windows"]} windows, '
      f'{len(d["trials"])} folds (stored; --full-eval to recompute)')
PY
  record 0 "encoder comparison (stored)"
fi

if [ "$MIC" -eq 1 ]; then
  step "9. MICROPHONE ACCEPTANCE (synchronisation gate)"
  rm -f /tmp/vs_timing.csv
  spawn ros2 run contact_audio_perception contact_ml_node --ros-args \
    -p model_path:="$SHARE/models/contact_model_cnn" >/tmp/vs_mlm.log 2>&1
  sleep 8
  spawn ros2 run contact_audio_perception audio_daq_node --ros-args \
    -p source:=mic -p publish_rate_hz:=15.0 \
    -p timing_log_path:=/tmp/vs_timing.csv >/tmp/vs_daqm.log 2>&1
  sleep 3
  # Measure the rate, do not read timeout's exit code: `timeout` returns 124 by
  # design here, which would report a healthy 15 Hz pipeline as a failure.
  timeout 12 ros2 topic hz /audio/contact_prediction --window 60 \
    >/tmp/vs_hz_mic.txt 2>&1
  python3 - <<'PY'
import re, sys
text = open('/tmp/vs_hz_mic.txt').read()
values = [float(v) for v in re.findall(r'average rate:\s*([\d.]+)', text)]
rate = max(values) if values else 0.0
print(f'  live prediction rate {rate:.2f} Hz (expect ~15 Hz)')
sys.exit(0 if rate >= 13.0 else 1)
PY
  CLASSIFY=$?
  python3 scripts/characterize_timing.py 70 --source mic --csv /tmp/vs_timing.csv \
    --require-predictions
  SYNC=$?
  cleanup
  echo
  echo "  CLASSIFICATION: $([ $CLASSIFY -eq 0 ] && echo PASS || echo FAIL)"
  echo "  SYNCHRONIZATION: $([ $SYNC -eq 0 ] && echo PASS || echo FAIL)"
  if [ $SYNC -ne 0 ]; then
    echo "  DO NOT RECORD SYNCHRONIZED BAGS ON THIS BACKEND"
    FAILURES+=("microphone synchronisation gate")
  fi
fi

printf '\n\033[1m== SUMMARY ==\033[0m\n'
echo "  commit $(git rev-parse --short HEAD)"
if [ ${#FAILURES[@]} -eq 0 ]; then
  echo "  ALL CHECKS PASSED"
  [ "$MIC" -eq 0 ] && \
    echo "  (microphone synchronisation NOT checked; run --mic on the lab machine)"
  STATUS=0
else
  echo "  FAILED: ${#FAILURES[@]}"
  for f in "${FAILURES[@]}"; do echo "    - $f"; done
  STATUS=1
fi
# Run cleanup explicitly and drop the trap first: an EXIT trap can otherwise
# replace the script's status with its own, which silently turned a failed
# synchronisation gate into exit 0.
trap - EXIT INT TERM
cleanup >/dev/null 2>&1
exit "$STATUS"
