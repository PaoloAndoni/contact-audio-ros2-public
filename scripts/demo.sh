#!/usr/bin/env bash
# End-to-end held-out demo: train a model on trials 001-003, then run the live
# ROS 2 pipeline on the UNSEEN trial-004 clips and print a majority-vote verdict.
set -e
source /opt/ros/jazzy/setup.bash
cd ~/audio-manipulation/ros2_ws
source install/setup.bash
set +e

pkill -f contact_audio_perception 2>/dev/null
pkill -f 'topic echo' 2>/dev/null
sleep 1

DATA="$HOME/train_data"
HELDOUT="$HOME/heldout_train"
MODEL="$HOME/heldout_model.npz"
CLASSES="no_contact slide_table tap_table"

# 1) Build a training set WITHOUT trial 004, and train a model that has never seen it.
rm -rf "$HELDOUT"
for cls in $CLASSES; do
  mkdir -p "$HELDOUT/$cls"
  for t in 001 002 003; do
    cp "$DATA/$cls/${cls}_${t}.wav" "$HELDOUT/$cls/" 2>/dev/null
  done
done
echo "=== training held-out model (trials 001-003 only) ==="
python3 scripts/train_centroids.py "$HELDOUT" "$MODEL" 2>&1 | grep -E "windows|saved"

# 2) Start the live perception nodes with the held-out model.
ros2 run contact_audio_perception audio_window_node \
  --ros-args -p window_sec:=1.0 -p hop_sec:=0.5 -p mono:=true >/tmp/window.log 2>&1 & W1=$!
ros2 run contact_audio_perception ast_encoder_node >/tmp/ast.log 2>&1 & W2=$!
ros2 run contact_audio_perception contact_classifier_node \
  --ros-args -p model_path:="$MODEL" >/tmp/clf.log 2>&1 & W3=$!

echo "=== waiting for pipeline... ==="
for _ in $(seq 1 45); do
  grep -q "AST model ready" /tmp/ast.log && grep -q "Classifying" /tmp/clf.log && break
  sleep 1
done

# 3) Play each UNSEEN trial-004 clip and report the live majority-vote verdict.
for cls in $CLASSES; do
  WAV="$DATA/$cls/${cls}_004.wav"
  timeout 15 ros2 topic echo /audio/contact_prediction --field label >/tmp/p.txt 2>&1 & EP=$!
  sleep 1
  timeout 15 ros2 run contact_audio_perception wav_player_node \
    --ros-args -p wav_path:="$WAV" -p loop:=false >/dev/null 2>&1
  sleep 2
  kill "$EP" 2>/dev/null; wait "$EP" 2>/dev/null
  counts=$(grep -vE '^$|^-' /tmp/p.txt | sort | uniq -c | tr -s ' ' | paste -sd '|')
  verdict=$(grep -vE '^$|^-' /tmp/p.txt | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
  echo "TRUE=$cls (unseen trial 004) | windows: ${counts:-none} | VERDICT: ${verdict:-none}"
done

kill "$W1" "$W2" "$W3" 2>/dev/null
echo "=== demo done ==="
