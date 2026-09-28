#!/bin/bash
# Survey an unknown ROS 2 machine. READ-ONLY: this script only listens. It
# publishes nothing, sends no command, and never touches the arm.
#
# Run this FIRST, before anything else, on the lab computer. It answers in about
# a minute the questions that otherwise cost half a session of asking and
# poking: which distro, which topics exist and at what rate, whether the arm
# publishes a base->tool transform at all, what the camera is called, whether a
# microphone is visible, and -- if audio and the robot are on different
# computers -- whether their clocks agree.
#
#   bash scripts/lab_survey.sh            survey, print, and save a report
#   bash scripts/lab_survey.sh --quick    skip the rate measurements
#
# The report is written to lab_survey_<host>_<date>.txt. Take it home: it is the
# record of what the lab machine actually looked like.
set -o pipefail

QUICK=0
[ "$1" = "--quick" ] && QUICK=1

REPORT="lab_survey_$(hostname)_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee "$REPORT") 2>&1

hr() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

hr "0. Who and where"
echo "host        : $(hostname)"
echo "date        : $(date -Is)"
echo "user        : $(whoami)"
echo "uname       : $(uname -srm)"
echo "os          : $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"

hr "1. ROS 2 environment"
echo "ROS_DISTRO      : ${ROS_DISTRO:-<not sourced!>}"
echo "RMW             : ${RMW_IMPLEMENTATION:-<default>}"
echo "ROS_DOMAIN_ID   : ${ROS_DOMAIN_ID:-0 (default)}"
echo "ROS_LOCALHOST_ONLY: ${ROS_LOCALHOST_ONLY:-<unset>}"
echo "python3         : $(python3 --version 2>&1)"
if [ -z "$ROS_DISTRO" ]; then
  echo
  echo "  ROS is not sourced. Find and source it before reading anything below:"
  ls -d /opt/ros/*/ 2>/dev/null | sed 's/^/    source /;s|/$|/setup.bash|'
  echo "  Nothing below this line is meaningful until you do."
fi
echo
echo "This matters because the workspace was built on Jazzy. A different distro"
echo "means rebuilding from source here, not copying an install/ directory."

hr "2. Topics on the wire"
echo "(type included: a name alone does not tell you what is in it)"
ros2 topic list -t 2>/dev/null | sed 's/^/  /' || echo "  ros2 topic list failed"

hr "3. The transforms the actions are derived from"
echo "/tf publishers:"
ros2 topic info /tf 2>/dev/null | sed 's/^/  /' || echo "  no /tf"
echo
echo "Frames currently in the tf tree:"
timeout 8 ros2 run tf2_ros tf2_echo --help >/dev/null 2>&1
if ros2 topic list 2>/dev/null | grep -qx /tf; then
  timeout 10 python3 - <<'PY'
import sys
try:
    import rclpy
    from tf2_ros import Buffer, TransformListener
except Exception as exc:                      # noqa: BLE001
    print(f'  could not import tf2_ros: {exc}')
    sys.exit(0)
rclpy.init()
node = rclpy.create_node('lab_survey_tf_listener')
buf = Buffer()
TransformListener(buf, node)
end = node.get_clock().now().nanoseconds + 5_000_000_000
while node.get_clock().now().nanoseconds < end:
    rclpy.spin_once(node, timeout_sec=0.1)
# all_frames_as_yaml() keys every frame by its CHILD name and records its
# parent as a property. The root of the tree -- which on an arm is exactly the
# base frame you are looking for -- is therefore never a key, only a value.
# Collecting keys alone reports "bases found: none" on a perfectly good tree.
frames = buf.all_frames_as_yaml() or ''
names = set()
try:
    import yaml
    tree = yaml.safe_load(frames) or {}
    for child, info in tree.items():
        names.add(str(child))
        if isinstance(info, dict) and info.get('parent'):
            names.add(str(info['parent']))
except Exception:                             # noqa: BLE001
    for line in frames.splitlines():
        if line and not line.startswith(' '):
            names.add(line.split(':')[0].strip())
        elif 'parent:' in line:
            names.add(line.split('parent:')[1].strip().strip("'\""))
names = sorted(n for n in names if n)
if names:
    print('  frames: ' + ', '.join(names))
else:
    print('  NO FRAMES. Either nothing publishes /tf, or not during these 5 s.')
print()
print('  Candidate base->tool pairs (what derive_actions.py needs):')
bases = [n for n in names if 'base' in n.lower() or 'world' in n.lower()]
tools = [n for n in names if any(k in n.lower() for k in
                                ('tool', 'tcp', 'ee', 'hand', 'grip', 'flange'))]
if bases and tools:
    for b in bases:
        for t in tools:
            try:
                buf.lookup_transform(b, t, rclpy.time.Time())
                print(f'    OK       {b} -> {t}')
            except Exception as exc:          # noqa: BLE001
                print(f'    no       {b} -> {t}  ({type(exc).__name__})')
else:
    print(f'    bases found: {bases or "none"}')
    print(f'    tools found: {tools or "none"}')
    print('    If one side is empty, ASK which frame is the end effector.')
rclpy.shutdown()
PY
else
  echo "  /tf is not being published. Ask whether the driver is running."
fi

hr "4. Joint states"
if ros2 topic list 2>/dev/null | grep -qx /joint_states; then
  timeout 6 ros2 topic echo /joint_states --once --field name 2>/dev/null \
    | sed 's/^/  /' || echo "  /joint_states exists but published nothing in 6 s"
else
  echo "  no /joint_states"
fi

hr "5. Cameras"
CAMS=$(ros2 topic list -t 2>/dev/null | grep -iE 'sensor_msgs/msg/(Image|CompressedImage)' | awk '{print $1}')
if [ -n "$CAMS" ]; then
  echo "$CAMS" | sed 's/^/  /'
else
  echo "  no image topics. A policy without vision is possible but ask first."
fi

hr "6. Microphones visible to THIS machine"
python3 - <<'PY'
try:
    import sounddevice as sd
except Exception as exc:                      # noqa: BLE001
    print(f'  sounddevice not installed here: {exc}')
    print('  pip install sounddevice  (or run the audio side on your laptop)')
else:
    try:
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f"  [{i}] {d['name']}  in={d['max_input_channels']} "
                      f"rate={d['default_samplerate']:.0f}")
    except Exception as exc:                  # noqa: BLE001
        print(f'  could not enumerate devices: {exc}')
PY
echo
echo "If your microphone is NOT listed, the audio side cannot run here, and you"
echo "are in the two-computer case below."

hr "7. Clocks -- only matters if audio and robot run on DIFFERENT computers"
echo "local time  : $(date -Is)"
echo "unix ns     : $(date +%s%N)"
timedatectl 2>/dev/null | sed 's/^/  /' || echo "  timedatectl unavailable"
chronyc tracking 2>/dev/null | head -5 | sed 's/^/  /'
echo
echo "Every header.stamp in this pipeline is a capture time on the clock of the"
echo "machine that captured it. Two machines whose clocks differ by 200 ms give"
echo "you a dataset misaligned by 200 ms, and NOTHING downstream will notice --"
echo "the numbers all look fine. Run this section on BOTH machines and compare"
echo "'unix ns' taken within a second of each other."
echo
echo "Preferred fix, in order:"
echo "  1. Plug the microphone into the robot's computer. One clock, no problem."
echo "  2. Same NTP server on both, verified here, and accept a few ms."
echo "  3. Tap the microphone hard against the moving tool at the start and end"
echo "     of EVERY recording. That spike is visible in the audio and in the"
echo "     arm's motion, so the offset stays recoverable offline. Do this even"
echo "     if 1 or 2 worked -- it costs two seconds and it has saved datasets."

hr "8. Rates of anything already publishing"
if [ "$QUICK" -eq 1 ]; then
  echo "  skipped (--quick)"
else
  for t in /tf /joint_states $CAMS; do
    [ -z "$t" ] && continue
    printf '  %-28s ' "$t"
    timeout 6 ros2 topic hz "$t" 2>/dev/null | grep -m1 average || echo "silent"
  done
fi

hr "9. Can this machine run the encoder?"
python3 - <<'PY'
for mod in ('numpy', 'torch', 'torchaudio', 'transformers', 'safetensors'):
    try:
        m = __import__(mod)
        print(f'  {mod:14s} {getattr(m, "__version__", "?")}')
    except Exception:                         # noqa: BLE001
        print(f'  {mod:14s} MISSING')
PY
echo
echo "torch missing here is survivable: record a bag with raw audio and derive"
echo "embeddings afterwards. torch missing is NOT a reason to record nothing."

hr "SUMMARY"
echo "Report saved to: $REPORT"
echo
echo "Decide these three before recording anything:"
echo "  1. One computer or two? (section 6 + 7)"
echo "  2. Does base->tool exist, and what are the frames called? (section 3)"
echo "  3. Is there a camera, and is it wanted in the observation? (section 5)"
echo
echo "Then run:  bash scripts/lab_record.sh --help"
