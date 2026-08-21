# Third-party notices

The joint impedance controller is adapted from `wuphilipp/gello_software` commit
`c3eea50be1c34e426ac94a0192d0d32bd1adc188`, which contains code from
Franka Robotics GmbH. The retained source-file headers specify the Apache
License 2.0 and their original copyright holders.

Controlled reset trajectories use `MotionGenerator` from the installed
`franka_example_controllers` package rather than carrying a local copy.

`robot_driven_control_node.cpp` is adapted from ros2_control's
`controller_manager/src/ros2_control_node.cpp` at tag `2.54.0`. The fixed-rate
sleep is disabled for real Franka hardware so the blocking robot read drives
the control cycle. ros2_control is distributed under the Apache License 2.0.
