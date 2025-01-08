#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
import numpy as np
from trajectory_msgs.msg import JointTrajectory

# 输入一条轨迹，利用前馈力矩+PD控制器跟踪轨迹

class ChinMujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_ros2_node')
        self.rbt = input("Enter the robot name: (chin/jaka)")
        if self.rbt == 'chin':
            self.n = 6
            self.xml_file = '/home/chenwh/ga_ddp/src/mujoco_publisher/xml/chin_crb7.xml'
            self.desired_position = [0.0] * self.n
            self.k_p = [400, 400, 400, 100, 25, 25]  # 比例增益
            self.k_d = 2*np.sqrt(self.k_p)  # 微分增益
        elif self.rbt == 'jaka':
            self.n = 6
            self.xml_file = '/home/chenwh/ga_ddp/src/mujoco_publisher/xml/jaka_zu12.xml'
            self.desired_position = [0.0, np.pi/2, 0, np.pi/2, 0, 0]
            self.k_p = [256, 400, 225, 50, 20, 5]  # 比例增益
            self.k_d = [51, 80, 44, 10, 4, 1]  # 微分增益
        else:
            raise ValueError("Invalid robot name. Please enter 'chin' or 'jaka'.")
            
        self.paused = False
        self.PublishJointStates = self.create_publisher(JointState,'/current_states',10)
        self.PublishMujocoSimClock = self.create_publisher(Clock,'/clock',10)
        self.create_subscription(JointTrajectory, '/joint_trajectory', self.trajectory_callback, 10)
        self.create_subscription(JointState, '/desired_states', self.target_callback, 10)
        self.trajectory = []
        self.current_trajectory_index = 0
        self.counter = 0
        self.target_reached = False
        self.desired_velocity = [0.0] * self.n
        self.feedforward_torque = [0.0] * self.n

    def trajectory_callback(self, msg):
        self.trajectory = msg
        self.current_trajectory_index = 0
        self.target_reached = False
        self.counter = 0

        # Find the closest point in the trajectory based on the current time
        current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        for i in range(len(self.trajectory.points)):
            point = self.trajectory.points[i]
            point_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 + point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            time_diff = abs(current_time - point_time)
            if time_diff <= 0.005:
                self.current_trajectory_index = i
                break

    def target_callback(self, msg):
        self.desired_position = msg.position
        self.desired_velocity = msg.velocity
        self.feedforward_torque = msg.effort

    def ChinMujocoSim(self):
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)

        # 设定模型关节角的初始值
        for i in range(self.n):
            data.joint(i).qpos[0] = self.desired_position[i]

        with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
            while 1:
                step_start = time.time()

                # 读取mujoco的关节信息，并上传至topic：/joint_states
                joint_state_msg = JointState()
                joint_state_msg.header.stamp = self.get_clock().now().to_msg()
                joint_state_msg.position = [data.qpos[i] for i in range(self.n)]
                joint_state_msg.velocity = [data.qvel[i] for i in range(self.n)]
                joint_state_msg.effort = [data.qfrc_actuator[i] for i in range(self.n)]
                self.PublishJointStates.publish(joint_state_msg)

                # 判断是否到达期望状态
                position_error = [self.desired_position[i] - joint_state_msg.position[i] for i in range(self.n)]
                velocity_error = [self.desired_velocity[i] - joint_state_msg.velocity[i] for i in range(self.n)]

                if np.linalg.norm(position_error) < 1e-3 and np.linalg.norm(velocity_error) < 1e-3:
                    self.target_reached = True

                if self.target_reached:
                    data.ctrl[:] = [
                        self.k_p[i] * position_error[i] + self.k_d[i] * velocity_error[i] + self.feedforward_torque[i]
                        for i in range(self.n)
                    ]
                else:
                    if self.current_trajectory_index < len(self.trajectory.points):
                        target = self.trajectory.points[self.current_trajectory_index]
                        position_error = [target.positions[i] - joint_state_msg.position[i] for i in range(self.n)]
                        velocity_error = [target.velocities[i] - joint_state_msg.velocity[i] for i in range(self.n)]
                        data.ctrl[:] = [
                            self.k_p[i] * position_error[i] + self.k_d[i] * velocity_error[i] + target.effort[i]
                            for i in range(self.n)
                        ]
                        self.counter += 1
                        if self.counter >= 10:
                            self.current_trajectory_index += 1
                            self.counter -= 10
                    else:
                        raise RuntimeError("Trajectory not received")

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