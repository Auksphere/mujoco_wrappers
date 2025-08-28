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
            # self.xml_file = '/home/chenwh/ga_ddp/src/mujoco_publisher/xml/chin_crb7.xml'
            self.xml_file = 'models/chin_crb7/scene.xml'
            self.desired_position = np.array([0.0] * self.n)
            self.k_p = np.array([400, 400, 400, 100, 25, 25])  # 比例增益
            self.k_d = 2*np.sqrt(self.k_p)  # 微分增益
        elif self.rbt == 'jaka':
            self.n = 6
            # self.xml_file = '/home/chenwh/ga_ddp/src/mujoco_publisher/xml/jaka_zu12.xml'
            self.xml_file = 'models/jaka_zu12/scene.xml'
            self.desired_position = np.array([0, np.pi/2, 0, np.pi/2, 0, 0])
            self.k_p = np.array([400, 400, 400, 50, 25, 5])  # 比例增益
            self.k_d = np.array([50, 50, 50, 3, 2, 1])  # 微分增益
        else:
            raise ValueError("Invalid robot name. Please enter 'chin' or 'jaka'.")
        
        self.paused = False
        self.teaching_mode = False
        self.just_exited_teach_mode = False
        self.get_logger().info("Press 'T' to toggle teach mode. Press SPACE to pause/resume.")

        self.PublishMujocoSimClock = self.create_publisher(Clock,'/clock',10)
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)  # 发布关节状态
        self.desired_velocity = np.array([0.0] * self.n)
        self.feedforward_torque = np.array([0.0] * self.n)
        self.csv_file = open('log/joint_angles.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'desired_position', 'actual_position'])

    def ChinMujocoSim(self):
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        
        data = mujoco.MjData(model)

        # 设定模型关节角的初始值
        data.qpos[:] = self.desired_position
        
        start_time = time.time()
        with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # 当从示教模式切换回保持模式时，更新目标位置
                if not self.teaching_mode and self.just_exited_teach_mode:
                    self.desired_position = data.qpos.copy()
                    self.get_logger().info("Teach Mode OFF: Holding new position.")
                    self.just_exited_teach_mode = False

                #读取mujoco的关节信息
                joint_state_msg = JointState()
                joint_state_msg.position = list(data.qpos)
                joint_state_msg.velocity = list(data.qvel)
                joint_state_msg.effort = list(data.qfrc_smooth)
                
                self.joint_state_publisher.publish(joint_state_msg) # 发布关节状态

                if self.teaching_mode:
                    # 示教模式：只进行重力补偿
                    data.ctrl[:] = data.qpos[:]
                    gravity_compensation = data.qfrc_bias[:]
                    data.qfrc_applied[:] = gravity_compensation

                else:
                    # 保持/跟踪模式：使用PD控制器
                    self.desired_position = [np.sin((time.time() - start_time)/6) for _ in range(self.n)]
                    self.desired_position[1] += np.pi/2
                    self.desired_position[3] += np.pi/2
                    position_error = self.desired_position - data.qpos
                    velocity_error = self.desired_velocity - data.qvel
                    
                    pd_torques = self.k_p * position_error + self.k_d * velocity_error
                    data.ctrl[:] = pd_torques + self.feedforward_torque
                    # data.ctrl[:] = self.desired_position  # 似乎使用这个效果是一样的, 因为actuator本身就是一个pd控制器
                    # 公式为: torque = gainprm * data.ctrl + biasprm[0] + biasprm[1] * qpos + biasprm[2] * qvel

                self.csv_writer.writerow([step_start, list(self.desired_position), list(data.qpos)])

                if not self.paused:
                    mujoco.mj_step(model, data)
                    viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                rclpy.spin_once(self, timeout_sec=0.0)

    def key_callback(self,keycode):
        key = chr(keycode)
        if key == ' ':
            self.paused = not self.paused
        elif key == 'T':
            self.teaching_mode = not self.teaching_mode
            if self.teaching_mode:
                self.get_logger().info("Teach Mode ON: Robot is compliant. Drag it with Ctrl + Right Mouse.")
            else:
                # 设置一个标志，让主循环来处理位置更新
                self.just_exited_teach_mode = True

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