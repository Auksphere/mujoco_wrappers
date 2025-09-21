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
import os
import sys
from scipy.linalg import expm

# Add GUFIC_mujoco path to import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from misc_func import initialize_trajectory, set_gains, vee_map, hat_map, adjoint_g_ed, adjoint_g_ed_dual, adjoint_g_ed_deriv

def adjoint_g(g):
    R = g[:3, :3]
    p = g[:3, 3]
    Ad_g = np.zeros((6, 6))
    Ad_g[:3, :3] = R
    Ad_g[3:, 3:] = R
    Ad_g[:3, 3:] = hat_map(p) @ R
    return Ad_g

class RobotState:
    def __init__(self, model, data, ee_name, robot_name):
        self.model = model
        self.data = data
        self.ee_name = ee_name
        self.robot_name = robot_name
        self.site_id = self.model.site(ee_name).id
        if self.robot_name == 'chin':
            self.ee_body_id = self.model.body('chin_end_effector_mount').id
        elif self.robot_name == 'jaka':
            self.ee_body_id = self.model.body('jaka_end_effector_mount').id
        else:
            raise ValueError("Invalid robot name for RobotState")
            
        self.Jp = np.zeros((3, self.model.nv))
        self.Jr = np.zeros((3, self.model.nv))

    def update(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        # Use site position for jacobian calculation, but body id
        mujoco.mj_jac(self.model, self.data, self.Jp, self.Jr, self.data.site_xpos[self.site_id], self.ee_body_id)

    def get_pose(self):
        p = self.data.site_xpos[self.site_id]
        R = self.data.site_xmat[self.site_id].reshape(3, 3)
        return p.copy(), R.copy()

    def get_body_jacobian(self):
        self.update()
        J = np.vstack((self.Jp, self.Jr))
        p, R = self.get_pose()
        # This Adjoint is for the body frame of the end-effector link, not the site frame.
        # For simplicity, we assume site and body frames are close enough for this transform.
        # A more accurate approach might be needed if the site has a significant offset.
        body_p = self.data.xpos[self.ee_body_id]
        body_R = self.data.xmat[self.ee_body_id].reshape(3,3)
        g_body = np.vstack((np.hstack((body_R, body_p.reshape(3,1))), [0,0,0,1]))
        
        Ad_g_inv = np.linalg.inv(adjoint_g(g_body))
        return Ad_g_inv @ J

    def get_body_ee_velocity(self):
        self.update()
        Jb = self.get_body_jacobian()
        return Jb @ self.data.qvel[:self.model.nv].reshape(-1, 1)

    def get_full_inertia(self):
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def get_bias_torque(self):
        return self.data.qfrc_bias[:self.model.nv]

    def get_ee_force(self):
        # Simplified force sensor reading
        if self.robot_name == 'chin':
            force_sensor_name = 'chin_force_sensor'
            torque_sensor_name = 'chin_torque_sensor'
        else: # jaka
            force_sensor_name = 'jaka_force_sensor'
            torque_sensor_name = 'jaka_torque_sensor'
            
        force = self.data.sensor(force_sensor_name).data.copy()
        torque = self.data.sensor(torque_sensor_name).data.copy()
        return np.hstack((force, torque)).reshape(6, 1), np.zeros((6, 1))

    def set_control_torque(self, tau, gripper_ctrl=None):
        self.data.ctrl[:self.model.nv] = tau
        if gripper_ctrl is not None:
            # Assuming 2 gripper actuators
            self.data.ctrl[self.model.nv:] = gripper_ctrl

    def update_dynamic(self):
        mujoco.mj_step(self.model, self.data)

class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_ros2_node')
        self.rbt = input("Enter the robot name: (chin/jaka)")
        # self.rbt = "chin"
        if self.rbt == 'chin':
            self.n = 6
            self.xml_file = 'models/chin_crb7/chin_wiping_surface.xml'
            # self.desired_position = [0.0] * self.n
            # self.desired_position = [0, 0.327, -1.83, -0.6, -1.57, -1.57]
            self.desired_position = [-0.25, 0.24, -1.8, -0.52, -1.57, -1.78]
            self.k_p = [400, 400, 400, 100, 25, 25]
            self.k_d = 2 * np.sqrt(self.k_p)
            self.ee_name = 'chin_end_effector'
        elif self.rbt == 'jaka':
            self.n = 6
            self.xml_file = 'models/jaka_zu12/jaka_wiping_surface.xml'
            self.desired_position = [-0.149, 1.51, -1.73, 1.8, 1.47, 2.97]
            # self.desired_position = [0.0] * self.n
            self.k_p = [400, 400, 400, 50, 25, 5]
            self.k_d = [5, 5, 5, 3, 2, 1]
            self.ee_name = 'jaka_end_effector'
        else:
            raise ValueError("Invalid robot name. Please enter 'chin' or 'jaka'.")
        

        self.paused = False
        self.PublishMujocoSimClock = self.create_publisher(Clock, '/clock', 10)
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)  # 发布关节状态
        # Create separate CSV files for different data
        self.joint_csv_file = open('log/joint_angles.csv', mode='w', newline='')
        self.joint_csv_writer = csv.writer(self.joint_csv_file)
        self.joint_csv_writer.writerow(['time', 'joint_positions'])
        
        # Create CSV file for trajectory tracking data
        self.trajectory_csv_file = open(f'log/trajectory_tracking.csv', mode='w', newline='')
        self.trajectory_csv_writer = csv.writer(self.trajectory_csv_file)
        self.trajectory_csv_writer.writerow(['time', 'desired_x', 'desired_y', 'desired_z', 
                                           'actual_x', 'actual_y', 'actual_z',
                                           'error_x', 'error_y', 'error_z', 'error_norm'])

        # GUFIC Initialization
        self.iter = 0
        self.dt = 0.001 # Should be set from model
        
        self.task = input("Enter the task: regulation/circle/line/sphere: ") # "regulation", "circle", "line", "sphere"
        
        if self.task == "sphere":
            if self.rbt == "jaka":
                self.xml_file = '/home/auksphere/robotics/mujoco_wrappers/models/jaka_zu12/jaka_wiping_sphere.xml'
            else:
                self.xml_file = '/home/auksphere/robotics/mujoco_wrappers/models/chin_crb7/chin_wiping_sphere.xml'

        self.pd_t, self.Rd_t, self.dpd_t, self.dRd_t, self.ddpd_t, self.ddRd_t = initialize_trajectory(self.task, name = self.rbt)
        self.Kp, self.KR, self.Kd, self.kp_force, self.kd_force, self.ki_force, self.zeta = set_gains("GUFIC", self.task, self.rbt)

        self.fz = 10.0
        self.int_sat = 50
        self.e_force_prev = np.zeros((6, 1))
        self.int_force_prev = np.zeros((6, 1))

        self.T_f_low, self.T_f_high, self.delta_f = 0.5, 20, 1
        self.T_i_low, self.T_i_high, self.delta_i = 0.5, 20, 1
        self.x_tf = np.sqrt(2 * 10)
        self.x_ti = np.sqrt(2 * 10)
        self.gd = np.eye(4)

        self.max_time = 10.0 # seconds

    def get_velocity_field(self, g, V, t):
        zeta = self.zeta
        pd = self.pd_t(t).reshape((-1,))
        Rd = self.Rd_t(t)
        dpd = self.dpd_t(t).reshape((-1,))
        dRd = self.dRd_t(t)
        ddpd = self.ddpd_t(t).reshape((-1,))
        ddRd = self.ddRd_t(t)

        p = g[:3, 3]
        R = g[:3, :3]
        v = V[:3]
        w = V[3:]

        vd_star = R.T @ dRd @ Rd.T @ (p - pd) + R.T @ dpd - zeta * R.T @ (p - pd)
        wd_star = vee_map(R.T @ dRd @ Rd.T @ R - zeta * (Rd.T @ R - R.T @ Rd)).reshape((-1,))
        Vd_star = np.hstack((vd_star, wd_star))

        term1 = -hat_map(w) @ R.T @ dRd @ Rd.T @ R + R.T @ ddRd @ Rd.T @ R + R.T @ dRd @ dRd.T @ R + R.T @ dRd @ Rd.T @ R @ hat_map(w)
        term2 = -hat_map(w) @ R.T @ dRd @ Rd.T @ (p - pd) + R.T @ ddRd @ Rd.T @ (p - pd) + R.T @ dRd @ dRd.T @ (p - pd) \
                + R.T @ dRd @ Rd.T @ (R.T @ v - pd) - hat_map(w) @ R.T @ dpd + R.T @ ddpd
        term3 = dRd.T @ R + Rd.T @ R @ hat_map(w) + hat_map(w) @ R.T @ Rd - R.T @ dRd
        term4 = - hat_map(w) @ R.T @ (p - pd) + v - R.T @ dpd
        dvd_star = term2 - zeta * term4
        dwd_star = vee_map(term1 - zeta * term3).reshape((-1,))
        dVd_star = np.hstack((dvd_star, dwd_star))

        return Vd_star, dVd_star

    def get_force_field(self, g, gd):
        return np.array([0, 0, self.fz, 0, 0, 0]).reshape((-1,1))

    def geometric_unified_force_impedance_control(self):
        t = self.iter * self.dt
        Jb = self.robot_state.get_body_jacobian()
        qfrc_bias = self.robot_state.get_bias_torque()
        M = self.robot_state.get_full_inertia()
        p, R = self.robot_state.get_pose()
        g = np.eye(4); g[:3, :3] = R; g[:3, 3] = p
        Vb = self.robot_state.get_body_ee_velocity()

        # Desired trajectory
        pd = self.pd_t(t).reshape(-1)
        Rd = self.Rd_t(t)
        dpd = self.dpd_t(t).reshape(-1)
        dRd = self.dRd_t(t)
        ddpd = self.ddpd_t(t).reshape(-1)
        ddRd = self.ddRd_t(t)

        # Velocity field
        Vd_star, dVd_star = self.get_velocity_field(g, Vb.reshape(-1), t)
        Vd_star = Vd_star.reshape(-1, 1)
        dVd_star = dVd_star.reshape(-1, 1)

        # Force field
        gd_bar = np.eye(4)
        gd_bar[:3, :3] = Rd
        gd_bar[:3, 3] = pd
        Fd_star = self.get_force_field(g, gd_bar)

        # Positional force
        fp = R.T @ Rd @ self.Kp @ Rd.T @ (p - pd).reshape(-1,1)
        fR = vee_map(self.KR @ Rd.T @ R - R.T @ Rd @ self.KR)
        fg = np.vstack((fp,fR))

        # Force control
        Fe, d_Fe = self.robot_state.get_ee_force()
        e_force = -Fe - Fd_star
        
        self.int_force_prev += e_force * self.dt
        self.int_force_prev = np.clip(self.int_force_prev, -self.int_sat, self.int_sat)
        
        F_f = self.kp_force * e_force + self.ki_force * self.int_force_prev

        # Energy tank for force control
        inner_product_f = (Vb.T @ F_f).item()
        if inner_product_f >= 0:
            gamma_f = 1.0
        else:
            gamma_f = np.exp(inner_product_f / self.delta_f)
        F_f_mod = gamma_f * F_f

        # Energy tank for impedance control
        inner_product_i = (Vd_star.T @ (F_f_mod + Fe)).item()
        if inner_product_i <= 0:
            gamma_i = 1.0
        else:
            gamma_i = np.exp(-inner_product_i / self.delta_i)
        
        Vd_star_mod = gamma_i * Vd_star
        dVd_star_mod = gamma_i * dVd_star
        
        ev_mod = Vb - Vd_star_mod

        # GUFIC control law
        M_tilde_inv = Jb @ np.linalg.pinv(M) @ Jb.T
        M_tilde = np.linalg.pinv(M_tilde_inv)
        
        tau_tilde = M_tilde @ (dVd_star_mod - M_tilde_inv @ self.Kd @ ev_mod - M_tilde_inv @ fg + M_tilde_inv @ F_f_mod)
        tau_cmd = Jb.T @ tau_tilde + qfrc_bias.reshape((-1, 1))

        return tau_cmd.flatten()

    def MujocoSim(self):
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)
        self.dt = model.opt.timestep

        self.robot_state = RobotState(model, data, self.ee_name, self.rbt)
        
        
            
        # Initialize trajectory history for visualization and analysis
        self.desired_trajectory_history = []
        self.actual_trajectory_history = []
        self.max_trajectory_points = 1000  # Maximum number of points to keep in history

        # Initialize robot to a default position if desired_position is defined
        try:
            for i in range(self.n):
                if hasattr(self, 'desired_position') and i < len(self.desired_position):
                    data.qpos[i] = self.desired_position[i]
        except:
            pass  # Use default initial position if desired_position is not available
        
        p_init, R_init = self.robot_state.get_pose()
        self.gd[:3,3] = p_init
        self.gd[:3,:3] = R_init

        with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
            while viewer.is_running():
                step_start = time.time()

                if not self.paused:
                    # Check if max time reached and exit if so
                    t = self.iter * self.dt
                    if t >= self.max_time:
                        print(f"Simulation completed. Max time {self.max_time}s reached.")
                        break
                    
                    # Print time step every 1000 iterations
                    if self.iter % 1000 == 0:
                        print(f"Time Step: {self.iter}")
                    
                    # Calculate desired trajectory position for visualization
                    t = self.iter * self.dt
                    pd_current = self.pd_t(t)
                    
                    # Get actual end-effector position
                    p_actual, R_actual = self.robot_state.get_pose()
                    
                    
                    # Calculate tracking error
                    if pd_current.ndim > 1:
                        pd_current_flat = pd_current.flatten()
                    else:
                        pd_current_flat = pd_current
                    
                    error = p_actual - pd_current_flat
                    error_norm = np.linalg.norm(error)
                    
                    # Record trajectory tracking data
                    self.trajectory_csv_writer.writerow([
                        t,  # time
                        pd_current_flat[0], pd_current_flat[1], pd_current_flat[2],  # desired position
                        p_actual[0], p_actual[1], p_actual[2],  # actual position
                        error[0], error[1], error[2],  # position error
                        error_norm  # error norm
                    ])
                    
                    tau_cmd = self.geometric_unified_force_impedance_control()
                    self.robot_state.set_control_torque(tau_cmd)
                    self.robot_state.update_dynamic()
                    viewer.sync()
                    self.iter += 1

                # Record joint angles
                joint_state_msg = JointState()
                joint_state_msg.position = [data.qpos[i] for i in range(self.n)]
                self.joint_state_publisher.publish(joint_state_msg)
                self.joint_csv_writer.writerow([self.iter * self.dt, joint_state_msg.position])

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                rclpy.spin_once(self, timeout_sec=0.0)

    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused

def main(args=None):
    rclpy.init(args=args)
    mujoco_node = MujocoNode()

    try:
        mujoco_node.MujocoSim()
    except KeyboardInterrupt:
        pass
    finally:
        mujoco_node.joint_csv_file.close()
        mujoco_node.trajectory_csv_file.close()
        
        mujoco_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
