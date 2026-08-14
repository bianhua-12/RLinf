#!/usr/bin/env bash
set -e

stop_process() {
  pkill -TERM -f "$1" || true
  for _ in {1..20}; do
    pgrep -f "$1" >/dev/null || return 0
    sleep 0.1
  done
  pkill -KILL -f "$1" || true
}

stop_process '[v]r_data_publisher.py'
stop_process '[R]oboticsServiceProcess'
rm -f /tmp/vr_data.ipc

bash /opt/apps/roboticsservice/runService.sh >/tmp/roboticsservice.log 2>&1
for _ in {1..100}; do
  ss -ltnH 'sport = :60061' | grep -q . && break
  sleep 0.1
done
ss -ltnH 'sport = :60061' | grep -q . || {
  echo "RoboticsServiceProcess failed to listen on port 60061" >&2
  exit 1
}

/data/pico_software/XRoboToolkit-Teleop-Sample-Python/.venv/bin/python - <<'PY'
import time
import xrobotoolkit_sdk as xrt

xrt.init()
try:
    first = xrt.get_time_stamp_ns()
    for _ in range(300):
        time.sleep(0.1)
        if xrt.get_time_stamp_ns() != first:
            break
    else:
        raise RuntimeError("PICO data timestamp did not advance within 30 seconds")
finally:
    xrt.close()
PY

exec /data/pico_software/XRoboToolkit-Teleop-Sample-Python/.venv/bin/python \
  /data/pico_software/vr_data_publisher.py \
  --config /data/pico_software/configs/vr_bridge.yaml
