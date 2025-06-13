# Robots Description (URDF & MJCF)

## Chin CRB7

<p float="left">
  <img src="ur10e.png" width="400">
</p>

## JAKA Zu 12

- ur10: This package contains a simplified robot description (MJCF) of the
[UR10](https://www.universal-robots.com/products/ur10-robot/) developed by
[Universal Robots](https://www.universal-robots.com/). It is derived from the
[publicly available URDF
description](https://github.com/JakaCobot/jaka_robot/tree/main/jaka_robot_v2.2/src/jaka_description/urdf).

<p float="left">
  <img src="ur10e.png" width="400">
</p>

## UR10

- ur10: This package contains a simplified robot description (MJCF) of the
[UR10](https://www.universal-robots.com/products/ur10-robot/) developed by
[Universal Robots](https://www.universal-robots.com/). It is derived from the
[publicly available URDF
description](https://github.com/ros-industrial/universal_robot/tree/noetic-devel/ur_description/urdf).

<p float="left">
  <img src="ur10e.png" width="400">
</p>

## Franka Emika Panda

- panda: This package contains a simplified robot description (MJCF) of the
[Franka Emika Panda](https://www.universal-robots.com/products/ur10-robot/) developed by
[Universal Robots](https://www.universal-robots.com/). It is derived from the
[publicly available URDF
description](https://github.com/frankaemika/franka_ros/tree/develop/franka_description/robots/panda).

<p float="left">
  <img src="ur10e.png" width="400">
</p>

## URDF → MJCF derivation steps

1. Loaded the URDF into MuJoCo and saved a corresponding MJCF.
2. Added position-controlled actuators and joint damping and armature. Note that these values have not been carefully tuned -- contributions are more than welcome to improve them.
3. Added home joint configuration as a `keyframe`.
4. Added `scene.xml`, which includes the robot, with a textured ground plane, skybox, and haze.