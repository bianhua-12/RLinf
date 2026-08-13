cd /data/RLinf-franka
source /data/RLinf/.venv/bin/activate
export RLINF_KEYBOARD_DEVICE=/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
bash examples/embodiment/collect_data.sh \
  realworld_dual_franka_collect_data_pico
