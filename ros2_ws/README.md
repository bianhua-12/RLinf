# RLinf Franka ROS 2 workspace

This workspace contains the RLinf-owned FR3 controller used by the real-world
GELLO collector. It builds against ROS 2 Humble and the system Franka ROS 2
installation; it does not require the GELLO repository at runtime.

```bash
source /opt/ros/humble/setup.bash
source /home/pnp/franka/franka_ros2_ws/install/setup.bash
cd /home/pnp/workspaces/RLinf/ros2_ws
colcon build --packages-select rlinf_franka_controller --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

The collector launches `single_fr3.launch.py` itself. Do not launch this
controller separately while collecting data.

The joint impedance controller configures Franka's collision behavior when it
starts. `cartesian_collision_threshold_scale` in
`src/rlinf_franka_controller/config/controllers.yaml` scales the official
Cartesian force/torque thresholds while keeping the joint collision thresholds
at the official defaults. The current `2.0` setting is intended for contact-rich
manipulation; keep the emergency stop accessible and lower the scale if
unintended contact is not detected early enough.

For real Franka hardware, this launch file uses RLinf's
`robot_driven_control_node`. Franka's blocking 1 kHz read supplies the control
loop timing, so the node does not add the fixed-rate sleep used by the stock
`ros2_control_node`. Fake hardware keeps the fixed-rate sleep because it has no
blocking robot read.

Verify the complete launch path without connecting to a robot:

```bash
ros2 launch rlinf_franka_controller single_fr3.launch.py \
  robot_ip:=127.0.0.1 namespace:=smoke arm_prefix:=smoke \
  use_fake_hardware:=true
```
