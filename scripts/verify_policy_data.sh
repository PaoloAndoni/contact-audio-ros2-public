#!/usr/bin/env bash
# One command for the three policy-data pieces: embedding, observation, action.
#
#   ./scripts/verify_policy_data.sh            full run (~3 minutes)
#   ./scripts/verify_policy_data.sh --quick    shorter collection windows
#
# Everything runs off a replayed WAV, a synthetic camera and fake_robot_state,
# so it needs no microphone, no camera and no robot. What it CANNOT check is
# stated at the end rather than quietly skipped.
#
# No `set -u`: ROS 2's setup.bash references unbound variables and aborts under it.
# `set -m` gives every background job its own process group so the whole group
# can be signalled by PID -- the same discipline verify_sprint.sh uses, and for
# the same reason: a surviving DAQ publishing onto the next step's topic corrupts
# every timestamp downstream.
set -o pipefail
set -m

QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS" || exit 1
RAW="${RAW_DIR:-$HOME/train_data}"
WAV="${WAV_PATH:-$RAW/tap_table/tap_table_001.wav}"
OUT="${OUT_DIR:-/tmp/policy_data}"
mkdir -p "$OUT"

if [ "$QUICK" -eq 1 ]; then EMB_SEC=12; BAG_SEC=25; else EMB_SEC=25; BAG_SEC=45; fi

CHILDREN=()
SELF_PGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')

kill_child() {
  local sig="$1" pid="$2"
  case "$pid" in ''|*[!0-9]*) return ;; esac
  [ "$pid" != "$SELF_PGID" ] && kill "-$sig" -- "-$pid" 2>/dev/null
  kill "-$sig" "$pid" 2>/dev/null
  return 0
}

cleanup() {
  local rc=$? pid
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

NODES='audio_daq_node|contact_ml_node|fake_robot_state|fake_camera|observation_node'
settle() {
  local waited=0
  while [ "$waited" -lt 15 ]; do
    if ! ros2 node list 2>/dev/null | grep -qE "$NODES"; then return 0; fi
    sleep 1
    waited=$((waited + 1))
  done
  # A node that outlives its stage is not cosmetic. It keeps publishing and
  # keeps competing for CPU, which slows the NEXT stage's encoder. That is how a
  # leaked audio_daq_node turns into "audio history spans 5.2 s" for every
  # candidate step and a derivation that emits nothing: the observation
  # assembler is correctly refusing a history stretched by dropped embeddings.
  # So kill the strays by name instead of warning and carrying on.
  echo "  WARNING: nodes still present after cleanup, killing strays:"
  ros2 node list 2>/dev/null | grep -E "$NODES" | sed 's/^/    /'
  local pid
  for pid in $(pgrep -f "$NODES" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    kill -TERM "$pid" 2>/dev/null
  done
  sleep 2
  for pid in $(pgrep -f "$NODES" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    kill -KILL "$pid" 2>/dev/null
  done
  sleep 2
  if ros2 node list 2>/dev/null | grep -qE "$NODES"; then
    echo "  STILL PRESENT. The next stage runs under extra load; treat its"
    echo "  rates and any span-related drops as measurements of THIS, not of"
    echo "  the pipeline:"
    ros2 node list 2>/dev/null | grep -E "$NODES" | sed 's/^/    /'
    return 1
  fi
  echo "  strays cleared."
  return 0
}

FAILURES=()
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
record() {
  if [ "$1" -eq 0 ]; then echo "  -> PASS: $2"; else echo "  -> FAIL: $2"; FAILURES+=("$2"); fi
}

source /opt/ros/jazzy/setup.bash

step "0. Build and unit tests"
colcon build 2>&1 | tail -2
record $? "build"
source install/setup.bash
colcon test --packages-select contact_audio_perception 2>&1 | tail -2
colcon test-result --all 2>&1 | tail -1
colcon test-result --all >/dev/null 2>&1
record $? "unit tests"
SHARE=$(python3 -c "from ament_index_python.packages import get_package_share_directory as g; print(g('contact_audio_perception'))")
# The CNN encoder, not AST: it sustains 15 Hz on this CPU, so the data path is
# exercised at the rate it is meant to run at rather than at AST's ~0.6 Hz.
MODEL="$SHARE/models/contact_model_cnn"
echo "  commit $(git rev-parse --short HEAD), model $(basename "$MODEL")"

if [ ! -f "$WAV" ]; then
  echo "  WAV not found: $WAV (set WAV_PATH=...)" >&2
  exit 2
fi

# ---------------------------------------------------------------- item 1 ----
step "1. Audio embedding published on its own topic"
spawn ros2 run contact_audio_perception contact_ml_node --ros-args \
  -p model_path:="$MODEL" >"$OUT/ml.log" 2>&1
sleep 8
spawn ros2 run contact_audio_perception audio_daq_node --ros-args \
  -p source:=wav -p wav_path:="$WAV" -p loop:=true -p publish_rate_hz:=15.0 \
  >"$OUT/daq.log" 2>&1
sleep 4
python3 scripts/verify_embedding.py "$EMB_SEC" --model "$MODEL" 2>&1 | tee "$OUT/embedding.txt"
record ${PIPESTATUS[0]} "embedding identity, provenance, content and rate"

echo
echo "  rate impact (the node reports what it achieves):"
timeout 10 ros2 topic hz /audio/spectrogram --window 60 >"$OUT/hz_spec.txt" 2>&1
timeout 10 ros2 topic hz /audio/embedding --window 60 >"$OUT/hz_emb.txt" 2>&1
timeout 10 ros2 topic hz /audio/contact_prediction --window 60 >"$OUT/hz_pred.txt" 2>&1
python3 - "$OUT" <<'PY'
import re, sys
def rate(name):
    text = open(f'{sys.argv[1]}/{name}').read()
    values = [float(m) for m in re.findall(r'average rate:\s*([\d.]+)', text)]
    return max(values) if values else 0.0
spec, emb, pred = rate('hz_spec.txt'), rate('hz_emb.txt'), rate('hz_pred.txt')
print(f'  spectrogram {spec:.2f} Hz | embedding {emb:.2f} Hz | prediction {pred:.2f} Hz')
# The claim is that publishing the embedding costs the classification path
# nothing. The prediction rate is the number that would move if it did.
sys.exit(0 if spec >= 13.0 and emb >= 13.0 and pred >= 13.0 else 1)
PY
record $? "embedding adds no measurable cost (all three topics still >= 13 Hz)"
grep -h 'embedding publish median' "$OUT/ml.log" | tail -2 | sed 's/^/  /'
cleanup; settle

# -------------------------------------------------------- items 2 and 3 ----
# Two tf rates. The residual error against the closed form should be dominated
# by interpolating a sinusoid from a finite sample rate, which falls as the
# square of the step. If it does NOT fall, the residual is a logic error rather
# than sampling -- so this is a check, not a flourish.
for TF_RATE in 100 400; do
  step "2. Record a synchronised bag (tf at ${TF_RATE} Hz)"
  BAG="$OUT/bag_${TF_RATE}"
  rm -rf "$BAG"
  spawn python3 scripts/fake_robot_state.py --rate "$TF_RATE" >"$OUT/robot.log" 2>&1
  spawn python3 scripts/fake_camera.py --rate 30 >"$OUT/camera.log" 2>&1
  spawn ros2 run contact_audio_perception contact_ml_node --ros-args \
    -p model_path:="$MODEL" >"$OUT/ml2.log" 2>&1
  sleep 8
  spawn ros2 run contact_audio_perception audio_daq_node --ros-args \
    -p source:=wav -p wav_path:="$WAV" -p loop:=true -p publish_rate_hz:=15.0 \
    >"$OUT/daq2.log" 2>&1
  sleep 3
  # observation_node runs alongside, so the LIVE path is exercised on the same
  # data the offline path is about to be built from.
  spawn ros2 run contact_audio_perception observation_node --ros-args \
    -p dump_path:="$OUT/live_${TF_RATE}.npz" >"$OUT/obs.log" 2>&1
  spawn timeout "$BAG_SEC" ros2 bag record -s mcap -o "$BAG" \
    /audio/spectrogram /audio/embedding /audio/contact_prediction \
    /camera/image_raw /tf /tf_static /joint_states >"$OUT/bag.log" 2>&1
  wait "${CHILDREN[-1]}" 2>/dev/null
  cleanup; settle
  [ -d "$BAG" ]
  record $? "bag recorded at tf ${TF_RATE} Hz"
  grep -h 'assembled' "$OUT/obs.log" | tail -2 | sed 's/^/  /'

  step "3. Derive (observation, action) pairs from the bag (tf ${TF_RATE} Hz)"
  # Delete the previous run's outputs FIRST. Without this, a derivation that
  # emits nothing leaves the earlier npz in place and steps 4-6 go on to verify
  # a dataset THIS RUN NEVER PRODUCED -- printing PASS, and feeding step 6 a
  # residual measured from different data. A stale artifact that still verifies
  # is worse than no artifact.
  rm -f "$OUT/actions_${TF_RATE}.npz" "$OUT/verify_obs_${TF_RATE}.txt" \
        "$OUT/verify_act_${TF_RATE}.txt"
  python3 scripts/derive_actions.py "$BAG" --out "$OUT/actions_${TF_RATE}.npz" \
    2>&1 | tee "$OUT/derive_${TF_RATE}.txt"
  DERIVED=${PIPESTATUS[0]}
  record $DERIVED "derivation at tf ${TF_RATE} Hz"

  if [ "$DERIVED" -ne 0 ] || [ ! -f "$OUT/actions_${TF_RATE}.npz" ]; then
    echo "  Steps 4 and 5 at ${TF_RATE} Hz cannot run: no dataset was produced."
    record 1 "observation assembly at tf ${TF_RATE} Hz (no dataset)"
    record 1 "action derivation at tf ${TF_RATE} Hz (no dataset)"
    continue
  fi

  step "4. Verify observations (tf ${TF_RATE} Hz)"
  python3 scripts/verify_observation.py "$OUT/actions_${TF_RATE}.npz" \
    2>&1 | tee "$OUT/verify_obs_${TF_RATE}.txt"
  record ${PIPESTATUS[0]} "observation assembly at tf ${TF_RATE} Hz"

  step "5. Verify actions against the closed form (tf ${TF_RATE} Hz)"
  python3 scripts/verify_actions.py "$OUT/actions_${TF_RATE}.npz" \
    2>&1 | tee "$OUT/verify_act_${TF_RATE}.txt"
  record ${PIPESTATUS[0]} "action derivation at tf ${TF_RATE} Hz"
done

step "6. Residual error is sampling, not misalignment"
python3 - "$OUT" <<'PY'
import re, sys
def worst(path, field):
    # Missing or unparsable means the stage that writes this file did not run.
    # Returning nan would make the comparison below silently false and report a
    # logic error that was never measured, so say so and stop instead.
    try:
        text = open(path).read()
    except OSError:
        print(f'  {path} is absent: its derivation produced no dataset, so the')
        print('  rate comparison has nothing to compare. Fix that stage first.')
        sys.exit(2)
    m = re.search(field + r'\s*([\d.eE+-]+)', text)
    if m is None:
        print(f'  no "{field}" found in {path}')
        sys.exit(2)
    return float(m.group(1))


# THE MEDIAN, NOT THE MAX. The max is an extreme value set by the widest /tf
# interpolation gap in the run, and wide gaps come from DROPPED transforms,
# which get more common as the publish rate goes up -- so the max can rise with
# the rate even while every ordinary sample improves. Measured here: the median
# fell 35x from 100 to 400 Hz while the max moved 0.8x, and at 400 Hz the worst
# gap was 16.4 ms against a nominal 2.5 ms. The max still has to clear the
# tolerance in step 5; that is its job. This step asks a different question.
# Measure the law on ONE interpolated pose. An action chunk aggregates 17
# poses, so one pose landing in a wide gap contaminates the whole chunk; that
# makes the action median improve only ~3x with the rate even though each pose
# improves ~35x. The chunk is the product, but the pose is where interpolation
# error actually enters, so it is what this claim is about.
POSE = r'pose_current median\s*:'
ACTION = r'translation median\s*:'
WORST = r'translation error\s*:\s*worst'
a = worst(f'{sys.argv[1]}/verify_act_100.txt', POSE)
b = worst(f'{sys.argv[1]}/verify_act_400.txt', POSE)
ca = worst(f'{sys.argv[1]}/verify_act_100.txt', ACTION)
cb = worst(f'{sys.argv[1]}/verify_act_400.txt', ACTION)
wa = worst(f'{sys.argv[1]}/verify_act_100.txt', WORST)
wb = worst(f'{sys.argv[1]}/verify_act_400.txt', WORST)
print(f'  median POSE error  : {a:.3e} m at 100 Hz, {b:.3e} m at 400 Hz')
print(f'  ratio {a / b:.1f}x (chord error of a sinusoid falls as the square of')
print('  the sample step, so 4x the rate should give roughly 16x less error.')
print('  A residual that does NOT fall would be a logic error, not sampling.)')
print(f'  median ACTION error: {ca:.3e} m at 100 Hz, {cb:.3e} m at 400 Hz '
      f'(ratio {ca / cb:.1f}x)')
print(f'  worst  ACTION error: {wa:.3e} m at 100 Hz, {wb:.3e} m at 400 Hz '
      f'(ratio {wa / wb:.1f}x)')
print('  The two ACTION rows do not follow the rate, and are not expected to:')
print('  a chunk spans 17 poses, so it inherits any dropped /tf in its span,')
print('  and drops get MORE common at a higher publish rate. Both stay far')
print('  under the 5e-05 m tolerance that step 5 enforces on every sample.')
sys.exit(0 if b < a / 4.0 else 1)
PY
record $? "interpolated pose error falls with the /tf sample rate"

printf '\n\033[1m== SUMMARY ==\033[0m\n'
echo "  commit $(git rev-parse --short HEAD)"
echo "  artifacts in $OUT"
if [ ${#FAILURES[@]} -eq 0 ]; then
  echo "  ALL CHECKS PASSED"
  STATUS=0
else
  echo "  FAILED: ${#FAILURES[@]}"
  for f in "${FAILURES[@]}"; do echo "    - $f"; done
  STATUS=1
fi
cat <<'NOTE'

  NOT CHECKED HERE, and not checkable without the lab:
    - a real camera's stamps (driver latency, exposure midpoint vs readout)
    - a real arm's /tf, including whether base->tool is published at all
    - whether the microphone's stamps are usable at all on the lab machine:
      verify_sprint.sh --mic is still the gate for that, and it FAILS on WSLg
    - the gripper channel: nothing here produces one, so it is ZERO, not measured
    - any claim about a trained policy. No model is trained and none is implied.
NOTE
trap - EXIT INT TERM
cleanup >/dev/null 2>&1
exit "$STATUS"
