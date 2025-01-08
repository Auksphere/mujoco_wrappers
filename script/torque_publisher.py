#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
import numpy as np

# 输入前馈力矩和期望位置，速度，计算输出力矩

joint_name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

class ChinMujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_ros2_node')
        self.paused = False

        self.PublishJointStates = self.create_publisher(JointState,'/joint_states',10)
        self.PublishMujocoSimClock = self.create_publisher(Clock,'/clock',10)
        self.k_p = [400, 400, 400, 100, 25, 25]  # 比例增益
        self.k_d = 2*np.sqrt(self.k_p)  # 微分增益
        self.create_subscription(JointState, '/joint_commands', self.joint_commands_callback, 10)
        self.desired_position = [0.0] * 6
        self.desired_velocity = [0.0] * 6
        self.feedforward_torque = [0.0] * 6
        self.xml_file = '/home/chenwh/ga_ddp/src/mujoco_publisher/xml/chin_crb7.xml'

    def joint_commands_callback(self, msg):
        self.desired_position = msg.position
        self.desired_velocity = msg.velocity
        self.feedforward_torque = msg.effort

    def ChinMujocoSim(self):
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)

        with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
            while 1:
                step_start = time.time()

                #读取mujoco的关节信息，并上传至topic：/joint_states
                joint_state_msg = JointState()
                joint_state_msg.name = joint_name
                joint_state_msg.position = [data.joint(i).qpos[0] for i in joint_name]
                joint_state_msg.velocity = [data.joint(i).qvel[0] for i in joint_name]
                joint_state_msg.effort = [data.joint(i).qfrc_smooth[0] for i in joint_name]

                self.PublishJointStates.publish(joint_state_msg)

                position_error = [self.desired_position[i] - joint_state_msg.position[i] for i in range(6)]
                velocity_error = [self.desired_velocity[i] - joint_state_msg.velocity[i] for i in range(6)]

                data.ctrl[:] = [
                    self.k_p[i] * position_error[i] + self.k_d[i] * velocity_error[i] + self.feedforward_torque[i]
                    for i in range(6)
                ]

                if not self.paused:
                    mujoco.mj_step(model, data)
                    viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                rclpy.spin_once(self, timeout_sec=0.0)

    def key_callback(self,keycode):
        if chr(keycode) == ' ':
          if self.paused == True:
              self.paused = False
          else :
              self.paused = True

def main(args=None):
    rclpy.init(args=args)
    chin_mujoco_node = ChinMujocoNode()

    try:
        chin_mujoco_node.ChinMujocoSim()
    except KeyboardInterrupt:
        pass
    finally:
        chin_mujoco_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()