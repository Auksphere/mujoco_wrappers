#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
import numpy as np
import csv

# 输入前馈力矩和期望位置，速度，计算输出力矩

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
            self.desired_position = [0, np.pi/2, 0, np.pi/2, 0, 0]
            self.k_p = [400, 400, 400, 50, 25, 5]  # 比例增益
            self.k_d = [5, 5, 5, 3, 2, 1]  # 微分增益
        else:
            raise ValueError("Invalid robot name. Please enter 'chin' or 'jaka'.")
        
        self.paused = False
        self.PublishMujocoSimClock = self.create_publisher(Clock,'/clock',10)
        self.desired_velocity = [0.0] * self.n
        self.feedforward_torque = [0.0] * self.n
        self.csv_file = open('/home/chenwh/ga_ddp/src/mujoco_publisher/log/joint_angles.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'desired_position', 'actual_position'])

    def ChinMujocoSim(self):
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)

        # 设定模型关节角的初始值
        for i in range(self.n):
            data.qpos[i] = self.desired_position[i]
        
        start_time = time.time()
        with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
            while 1:
                step_start = time.time()

                #读取mujoco的关节信息
                joint_state_msg = JointState()
                joint_state_msg.position = [data.qpos[i] for i in range(self.n)]
                joint_state_msg.velocity = [data.qvel[i] for i in range(self.n)]
                joint_state_msg.effort = [data.qfrc_smooth[i] for i in range(self.n)]
                
                #self.desired_position[0] = np.sin((time.time() - start_time)/6)
                position_error = [np.sin((time.time() - start_time)/6) + self.desired_position[i] - joint_state_msg.position[i] for i in range(self.n)]
                print(np.linalg.norm(position_error))
                velocity_error = [self.desired_velocity[i] - joint_state_msg.velocity[i] for i in range(self.n)]
                data.ctrl[:] = [
                    self.k_p[i] * position_error[i] + self.k_d[i] * velocity_error[i] + self.feedforward_torque[i]
                    for i in range(self.n)
                ]

                self.csv_writer.writerow([step_start, self.desired_position, joint_state_msg.position])

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
        chin_mujoco_node.csv_file.close()
        chin_mujoco_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()