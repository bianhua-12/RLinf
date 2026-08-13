#!/usr/bin/env bash
pkill -TERM -f '[v]r_data_publisher.py' || true
sleep 1
exec /data/pico_software/XRoboToolkit-Teleop-Sample-Python/.venv/bin/python \
  /data/pico_software/vr_data_publisher.py \
  --config /data/pico_software/configs/vr_bridge.yaml
