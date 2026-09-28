#!/bin/bash
# Record a synchronised bag in the lab, and say honestly what landed in it.
#
# The one rule this script is built around: RAW AUDIO IS RECORDED WHENEVER IT
# CAN BE. Spectrograms, embeddings and observations can all be recomputed from
# raw audio afterwards, with a different window length, a different encoder or
# a different normalisation. Raw audio cannot be recovered from any of them. A
# bag with raw audio keeps every choice open; a bag without it freezes tonight's
# choices into the only dataset you will ever have from this robot.
#
#   bash scripts/lab_record.sh --check                 is the mic usable here?
#   bash scripts/lab_record.sh --name demo_01          record (default 60 s)
#   bash scripts/lab_record.sh --name demo_01 --sec 90
#   bash scripts/lab_record.sh --name demo_01 --no-pipeline    raw audio only
#   bash scripts/lab_record.sh --inspect bags/demo_01          re-print a report
#
# Options:
#   --device D    microphone (index or name); default is the system default
#   --no-pipeline skip the encoder; record raw audio + robot only
#   --no-audio    robot only, when the microphone is dead
set -m
set -o pipefail

NAME=""; SEC=60; DEVICE=""; PIPELINE=1; AUDIO=1; CHECK=0; INSPECT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --sec) SEC="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --no-pipeline) PIPELINE=0; shift ;;
    --no-audio) AUDIO=0; PIPELINE=0; shift ;;
    --check) CHECK=1; shift ;;
    --inspect) INSPECT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

CHILDREN=()
cleanup() {
  local rc=$? pid
  for pid in "${CHILDREN[@]}"; do
    case "$pid" in ''|*[!0-9]*) continue ;; esac
    kill -TERM -- "-$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
  done
  sleep 1
  for pid in "${CHILDREN[@]}"; do
    case "$pid" in ''|*[!0-9]*) continue ;; esac
    kill -KILL -- "-$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
  done
  CHILDREN=()
  return $rc
}
trap cleanup EXIT INT TERM
spawn() { "$@" & CHILDREN+=($!); }
hr() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

if [ -z "$ROS_DISTRO" ]; then
  echo "ROS is not sourced. source /opt/ros/<distro>/setup.bash first." >&2
  exit 1
fi
[ -f install/setup.bash ] && source install/setup.bash

# ---------------------------------------------------------------- inspect --
inspect_bag() {
  local bag="$1"
  hr "What is actually in $bag"
  ros2 bag info "$bag" 2>/dev/null | sed 's/^/  /'
  python3 - "$bag" <<'PY'
import sys
from pathlib import Path
try:
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
except Exception as exc:                      # noqa: BLE001
    print(f'  rosbag2_py unavailable ({exc}); the ros2 bag info above is all.')
    sys.exit(0)
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag = sys.argv[1]
reader = SequentialReader()
reader.open(StorageOptions(uri=bag, storage_id=''),
            ConverterOptions('', ''))
types = {t.name: t.type for t in reader.get_all_topics_and_types()}
stamps = {}
recv = {}
frames = {}
while reader.has_next():
    topic, data, t_recv = reader.read_next()
    try:
        msg = deserialize_message(data, get_message(types[topic]))
    except Exception:                         # noqa: BLE001
        continue
    # TFMessage carries a LIST of TransformStamped and no header of its own,
    # so looking only for msg.header silently omits /tf -- the one stream every
    # action in this dataset is derived from. Take the first transform's stamp.
    hdr = getattr(msg, 'header', None)
    if hdr is None:
        transforms = getattr(msg, 'transforms', None)
        if transforms:
            hdr = transforms[0].header
            frames.setdefault(topic, set()).update(
                f'{t.header.frame_id}->{t.child_frame_id}' for t in transforms)
    if hdr is not None:
        stamps.setdefault(topic, []).append(
            hdr.stamp.sec * 10**9 + hdr.stamp.nanosec)
        recv.setdefault(topic, []).append(t_recv)

print()
print('  topic                          msgs   header rate   stamp-vs-write')
print('  ' + '-' * 68)
for topic in sorted(stamps):
    s, r = stamps[topic], recv[topic]
    n = len(s)
    span = (s[-1] - s[0]) / 1e9 if n > 1 else 0.0
    rate = (n - 1) / span if span > 0 else 0.0
    lag = [(b - a) / 1e6 for a, b in zip(s, r)]
    lag.sort()
    med = lag[len(lag) // 2] if lag else 0.0
    print(f'  {topic:28s} {n:6d}  {rate:8.2f} Hz  {med:+9.1f} ms')
print()
print('  stamp-vs-write is (time written to the bag) - (header.stamp).')
print('  It should be POSITIVE and roughly the pipeline latency. NEGATIVE means')
print('  a message claims to have been captured after it was written, which is')
print('  a broken clock or a re-stamped header -- do not trust that stream.')
for topic, pairs in sorted(frames.items()):
    print()
    print(f'  transforms on {topic}:')
    for pair in sorted(pairs):
        print(f'    {pair}')
    print('  derive_actions.py needs ONE of these to be base->end-effector.')
missing = [t for t in ('/tf', '/joint_states') if t not in stamps]
if missing:
    print()
    print(f'  MISSING, and the dataset needs them: {", ".join(missing)}')
audio = [t for t in stamps if t.startswith('/audio/')]
if not audio:
    print()
    print('  NO AUDIO IN THIS BAG. It records the robot only. That is a valid')
    print('  fallback, but it cannot answer anything about contact sound.')
PY
}

if [ -n "$INSPECT" ]; then
  inspect_bag "$INSPECT"
  exit 0
fi

# ------------------------------------------------------------------ check --
if [ "$CHECK" -eq 1 ]; then
  hr "Is the microphone usable on THIS machine?"
  echo "Listening for 25 s. Tap the microphone a few times."
  echo
  ARGS=(-p source:=mic -p publish_rate_hz:=15.0)
  [ -n "$DEVICE" ] && ARGS+=(-p device:="$DEVICE")
  spawn ros2 run contact_audio_perception audio_daq_node --ros-args "${ARGS[@]}" \
    >/tmp/lab_check.log 2>&1
  sleep 25
  cleanup
  echo "--- what the node reported ---"
  grep -iE 'window age|drift|device|rate|error|warn' /tmp/lab_check.log \
    | tail -20 | sed 's/^/  /'
  echo
  echo "READ THE 'window age' NUMBER. It must be POSITIVE and roughly the"
  echo "buffer length (~1000 ms). Negative means the capture timestamps are"
  echo "unusable on this machine -- which is exactly how it fails under WSLg."
  echo "If it is negative here too, record RAW AUDIO ONLY (--no-pipeline) and"
  echo "treat the audio/robot alignment as unresolved rather than pretending."
  echo
  echo "The fuller gate, if you have five minutes:  bash scripts/verify_sprint.sh --mic"
  exit 0
fi

# ----------------------------------------------------------------- record --
if [ -z "$NAME" ]; then
  echo "give the recording a name: --name demo_01" >&2
  exit 2
fi
mkdir -p bags
BAG="bags/$NAME"
if [ -e "$BAG" ]; then
  echo "$BAG already exists. Pick another name rather than overwriting a take." >&2
  exit 2
fi

TOPICS=(/tf /tf_static /joint_states)
for t in $(ros2 topic list -t 2>/dev/null \
           | grep -iE 'sensor_msgs/msg/(Image|CompressedImage)' | awk '{print $1}'); do
  TOPICS+=("$t")
done
if [ "$AUDIO" -eq 1 ]; then
  TOPICS+=(/audio/raw_chunk)
  [ "$PIPELINE" -eq 1 ] && TOPICS+=(/audio/spectrogram /audio/embedding /audio/contact_prediction)
fi

hr "About to record"
echo "  bag      : $BAG"
echo "  seconds  : $SEC"
echo "  topics   : ${TOPICS[*]}"
echo
echo "  SLATE THE TAKE. Tap the microphone hard against the tool once now and"
echo "  once at the end. That spike appears in the audio and in the arm's"
echo "  motion, so if the clocks turn out to disagree the offset is still"
echo "  recoverable afterwards. Two seconds of work; it has saved datasets."
echo
sleep 3

if [ "$AUDIO" -eq 1 ]; then
  ARGS=(-p sample_rate:=48000 -p channels:=1)
  [ -n "$DEVICE" ] && ARGS+=(-p device:="$DEVICE")
  spawn ros2 run contact_audio_perception audio_capture_node --ros-args "${ARGS[@]}" \
    >"/tmp/lab_capture_$NAME.log" 2>&1
  sleep 3
fi
if [ "$PIPELINE" -eq 1 ]; then
  SHARE=$(python3 -c "from ament_index_python.packages import get_package_share_directory as g; print(g('contact_audio_perception'))" 2>/dev/null)
  MODEL="$SHARE/models/contact_model_cnn"
  if [ -d "$MODEL" ]; then
    DAQ=(-p source:=mic -p publish_rate_hz:=15.0)
    [ -n "$DEVICE" ] && DAQ+=(-p device:="$DEVICE")
    spawn ros2 run contact_audio_perception audio_daq_node --ros-args "${DAQ[@]}" \
      >"/tmp/lab_daq_$NAME.log" 2>&1
    spawn ros2 run contact_audio_perception contact_ml_node --ros-args \
      -p model_path:="$MODEL" >"/tmp/lab_ml_$NAME.log" 2>&1
    sleep 6
  else
    echo "  no model bundle at $MODEL; recording raw audio only."
    TOPICS=("${TOPICS[@]/\/audio\/spectrogram}")
    TOPICS=("${TOPICS[@]/\/audio\/embedding}")
    TOPICS=("${TOPICS[@]/\/audio\/contact_prediction}")
  fi
fi

hr "Recording for $SEC s"
timeout "$SEC" ros2 bag record -s mcap -o "$BAG" "${TOPICS[@]}" 2>&1 | tail -3
cleanup

if [ ! -d "$BAG" ]; then
  echo "NO BAG WAS WRITTEN. Check the topic names above against 'ros2 topic list'." >&2
  exit 1
fi
inspect_bag "$BAG"

hr "Next"
echo "  Another take:   bash scripts/lab_record.sh --name ${NAME%_*}_$(printf '%02d' $(( 10#${NAME##*_} + 1 )) 2>/dev/null || echo XX)"
echo "  Derive pairs:   python3 scripts/derive_actions.py $BAG --out ${BAG}.npz"
echo "  Check them:     python3 scripts/verify_observation.py ${BAG}.npz"
echo
echo "  Copy the bags OFF this machine before you leave. Everything else can be"
echo "  redone from home; the recordings cannot."
