# Robots Description (URDF & MJCF)

## UR10

- ur10: This package contains a simplified robot description of the [UR10](https://www.universal-robots.com/products/ur10-robot/) developed by [Universal Robots](https://www.universal-robots.com/). It is derived from the [publicly available URDF description](https://github.com/ros-industrial/universal_robot/tree/noetic-devel/ur_description/urdf).
- The robot uses a position controller (to match the real hardware controller), and the PD parameters are manually specified and not tuned.

## Chin CRB7

- chin_crb7: This package contains a simplified robot description of the Chin CRB7 developed by Chin Robot. It is derived from the URDF description provided by hmbai
- The robot uses a torque controller (to match the real hardware controller).

## JAKA Zu 12

- jaka_zu12: This package contains a simplified robot description of the [JAKA Zu 12](https://www.jaka.com/en/productDetails/JAKA_Zu_12) developed by [JAKA Robotics](https://www.jaka.com/en/index). It is derived from the [publicly available URDF description](https://github.com/JakaCobot/jaka_robot/tree/main/jaka_robot_v2.2/src/jaka_description/urdf).
- The robot uses a position controller (to match the real hardware controller), and the PD parameters are manually specified and not tuned.

## Franka Emika Panda

- franka_panda: This package contains a simplified robot description of the [Franka Emika Panda](https://www.franka.de/) developed by [Franka Emika](https://www.franka.de/company). It is derived from the [publicly available URDF description](https://github.com/frankaemika/franka_ros/tree/develop/franka_description/robots/panda).
- The robot uses a torque controller (to match the real hardware controller).

## URDF → MJCF derivation steps

1. Loaded the URDF into MuJoCo and saved a corresponding MJCF.
2. Added position-controlled actuators and joint damping and armature. Note that these values have not been carefully tuned -- contributions are more than welcome to improve them.
3. Added home joint configuration as a `keyframe`.
4. Added `scene.xml`, which includes the robot, with a textured ground plane, skybox, and haze.