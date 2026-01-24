#!/usr/bin/env python3
"""
Asynchronous Admittance Control with Decoupled Policy and Controller Threads

Key Features:
- Asynchronous policy computation (25Hz) - computes IK solutions independently
- High-frequency controller (125Hz) - updates control commands using interpolated values
- Non-blocking simulation loop - maintains real-time performance
- Thread-safe linear interpolation between policy outputs

Architecture:
1. Policy Thread (25Hz): Computes inverse kinematics solutions asynchronously
2. Controller Thread (125Hz): Interpolates between IK solutions and sends control commands  
3. Simulation Loop: Runs MuJoCo physics continuously without blocking

This completely decouples computation from execution, preventing simulation stalls.
"""

import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
import numpy as np
import sys
import os
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from misc_func import calculate_desired_pose_trajectory, vee_map, hat_map, adjoint_g_ed, adjoint_g_ed_dual, adjoint_g
from filter import ButterLowPass
from scipy.linalg import block_diag, expm
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time

# Add project root to Python path to allow importing from 'controllers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import importlib.util

# First load util module
spec_util = importlib.util.spec_from_file_location("util", os.path.join(os.path.dirname(__file__), '..', '..', 'controllers', 'util.py'))
util_module = importlib.util.module_from_spec(spec_util)
sys.modules['controllers.util'] = util_module
spec_util.loader.exec_module(util_module)

# Then load ik_arm module
spec = importlib.util.spec_from_file_location("controllers.ik_arm", os.path.join(os.path.dirname(__file__), '..', '..', 'controllers', 'ik_arm.py'))
ik_arm_module = importlib.util.module_from_spec(spec)
sys.modules['controllers.ik_arm'] = ik_arm_module
spec.loader.exec_module(ik_arm_module)
IKArm = ik_arm_module.IKArm

@dataclass
class PolicyOutput:
    """Data structure for policy (IK computation) output"""
    timestamp: float
    desired_q: np.ndarray
    pd_modified: np.ndarray
    Rd_modified: np.ndarray
    x_admittance: np.ndarray
    success: bool

@dataclass
class RobotStateSnapshot:
    """Thread-safe snapshot of robot state data"""
    timestamp: float
    q: np.ndarray
    current_position: np.ndarray
    current_rotation: np.ndarray
    jacobian: np.ndarray
    force_torque: np.ndarray

class PolicyRequestQueue:
    """Thread-safe queue for robot state snapshots"""
    
    def __init__(self, max_size: int = 5):
        self.queue = queue.Queue(maxsize=max_size)
        
    def put_nowait(self, robot_state: RobotStateSnapshot):
        """Add a robot state snapshot (non-blocking)"""
        try:
            self.queue.put(robot_state, block=False)
        except queue.Full:
            # Skip if queue is full (policy computation is falling behind)
            pass
            
    def get(self, timeout: float = 0.1) -> Optional[RobotStateSnapshot]:
        """Get a robot state snapshot"""
        try:
            return self.queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

class LinearInterpolator:
    """Thread-safe linear interpolator for policy outputs"""
    
    def __init__(self, max_buffer_size: int = 10):
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.lock = threading.Lock()
        
    def add_sample(self, policy_output: PolicyOutput):
        """Add a new policy output to the buffer (thread-safe)"""
        with self.lock:
            self.buffer.append(policy_output)
            # Keep buffer size limited
            if len(self.buffer) > self.max_buffer_size:
                self.buffer.pop(0)
                
    def interpolate(self, target_time: float) -> Tuple[np.ndarray, Optional[dict]]:
        """Interpolate joint angles and pose data for target time (thread-safe)"""
        with self.lock:
            if len(self.buffer) == 0:
                return np.zeros(6), None
                
            if len(self.buffer) == 1:
                sample = self.buffer[0]
                return sample.desired_q.copy(), {
                    'pd_modified': sample.pd_modified.copy(),
                    'Rd_modified': sample.Rd_modified.copy(), 
                    'x_admittance': sample.x_admittance.copy()
                }
                
            # Find interpolation interval
            times = [sample.timestamp for sample in self.buffer]
            
            if target_time <= times[0]:
                sample = self.buffer[0]
                return sample.desired_q.copy(), {
                    'pd_modified': sample.pd_modified.copy(),
                    'Rd_modified': sample.Rd_modified.copy(),
                    'x_admittance': sample.x_admittance.copy()
                }
            elif target_time >= times[-1]:
                sample = self.buffer[-1] 
                return sample.desired_q.copy(), {
                    'pd_modified': sample.pd_modified.copy(),
                    'Rd_modified': sample.Rd_modified.copy(),
                    'x_admittance': sample.x_admittance.copy()
                }
            else:
                # Linear interpolation between two samples
                idx = 0
                for i in range(len(times) - 1):
                    if times[i] <= target_time <= times[i + 1]:
                        idx = i
                        break
                        
                t1, t2 = times[idx], times[idx + 1]
                alpha = (target_time - t1) / (t2 - t1)
                
                sample1, sample2 = self.buffer[idx], self.buffer[idx + 1]
                
                # Linear interpolation for joint angles
                interpolated_q = (1 - alpha) * sample1.desired_q + alpha * sample2.desired_q
                
                # Interpolate pose data
                pd_interp = (1 - alpha) * sample1.pd_modified + alpha * sample2.pd_modified
                x_interp = (1 - alpha) * sample1.x_admittance + alpha * sample2.x_admittance
                
                # SLERP for rotation matrices
                R_interp = self._slerp_rotation(sample1.Rd_modified, sample2.Rd_modified, alpha)
                
                return interpolated_q, {
                    'pd_modified': pd_interp,
                    'Rd_modified': R_interp,
                    'x_admittance': x_interp
                }
                
    def _slerp_rotation(self, R1, R2, t):
        """Spherical linear interpolation for rotation matrices"""
        r1 = ScipyRotation.from_matrix(R1)
        r2 = ScipyRotation.from_matrix(R2)
        key_rots = ScipyRotation.concatenate([r1, r2])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        return slerp(t).as_matrix()

class RobotState:
    def __init__(self, model, data, ee_name, robot_name):
        self.model = model
        self.data = data
        self.ee_name = ee_name
        self.robot_name = robot_name
        self.site_id = self.model.site(ee_name).id
        self.ee_body_id = self.model.body('jaka_end_effector_mount').id
            
        self.Jp = np.zeros((3, self.model.nv))
        self.Jr = np.zeros((3, self.model.nv))
        
        # Initialize low pass filter for force sensor
        dt = model.opt.timestep
        fs = 1 / dt
        cutoff = 10
        self.lp_filter = ButterLowPass(cutoff, fs, order=5)
        
        # Initialize state-space filter
        cut_off_freq = 5
        self.Ad, self.Bd = self.define_filter(cut_off_freq, dt)
        self.filter_state = np.zeros((12, 1))

    def define_filter(self, cutoff, dt, dim=6):
        ws = cutoff
        A = np.array([[0, 1],
                      [-ws**2, -2 * 1 * ws]])
        B = np.array([[0], [ws**2]])
        
        Ad1 = expm(A * dt)
        
        if np.linalg.det(A) != 0:
            Bd1 = np.linalg.inv(A) @ (Ad1 - np.eye(2)) @ B
        else:
            Bd1 = B * dt

        Ad = block_diag(*[Ad1 for _ in range(dim)])
        Bd = block_diag(*[Bd1 for _ in range(dim)])
        return Ad, Bd
    
    def lp_filter_implemented(self, force_torque):
        xf = self.filter_state[::2]
        dxf = self.filter_state[1::2]
        self.filter_state = self.Ad @ self.filter_state + self.Bd @ force_torque.reshape((-1,1))
        return xf, dxf

    def update(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jac(self.model, self.data, self.Jp, self.Jr, self.data.site_xpos[self.site_id], self.ee_body_id)

    def get_pose(self):
        p = self.data.site_xpos[self.site_id]
        R = self.data.site_xmat[self.site_id].reshape(3, 3)
        return p.copy(), R.copy()

    def get_body_jacobian(self):
        self.update()
        J = np.vstack((self.Jp, self.Jr))
        body_p = self.data.xpos[self.ee_body_id]
        body_R = self.data.xmat[self.ee_body_id].reshape(3,3)
        g_body = np.vstack((np.hstack((body_R, body_p.reshape(3,1))), [0,0,0,1]))
        Ad_g_inv = np.linalg.inv(adjoint_g(g_body))
        return Ad_g_inv @ J

    def get_ee_force(self):
        try:
            fx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor_fx")
            fy_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor_fy") 
            fz_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor_fz")
            
            fx = self.data.sensordata[self.model.sensor_adr[fx_id]]
            fy = self.data.sensordata[self.model.sensor_adr[fy_id]]
            fz = self.data.sensordata[self.model.sensor_adr[fz_id]]
            force = np.array([fx, fy, fz])
            
            mx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor_mx")
            my_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor_my")
            mz_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor_mz")
            
            mx = self.data.sensordata[self.model.sensor_adr[mx_id]]
            my = self.data.sensordata[self.model.sensor_adr[my_id]]
            mz = self.data.sensordata[self.model.sensor_adr[mz_id]]
            torque = np.array([mx, my, mz])
            
        except:
            print("Warning: Using legacy force sensors")
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor")
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            force = np.copy(self.data.sensordata[adr:adr + dim])
            
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor")
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            torque = np.copy(self.data.sensordata[adr:adr + dim])

        force_torque = np.concatenate([force, torque])
        ft, dft = self.lp_filter_implemented(force_torque)
        return ft, dft

class MujocoNode(Node):
    def __init__(self, task='sphere'):
        super().__init__('mujoco_controller_node')
        self.n = 6
        self.xml_file = 'models/jaka_zu12/jaka_admittance.xml'
        self.task = task
        
        # Trajectory parameters
        if task == 'circle':
            self.duration = 30.0
        else:
            self.duration = 10.0
            
        # Frequency settings
        self.policy_frequency = 25.0  # Policy (IK computation): 25Hz
        self.controller_frequency = 125.0  # Controller: 125Hz  
        self.policy_timestep = 1.0 / self.policy_frequency
        self.controller_timestep = 1.0 / self.controller_frequency
        
        # Thread control and communication
        self.policy_thread = None
        self.controller_thread = None
        self.simulation_running = False
        self.policy_running = False
        self.controller_running = False
        
        # Thread-safe communication
        self.policy_request_queue = PolicyRequestQueue(max_size=5)
        self.interpolator = LinearInterpolator(max_buffer_size=10)
        
        # Simulation state (shared between threads with locks)
        self.current_time = 0.0
        self.time_lock = threading.Lock()
        self.paused = False

        # Control parameters (accessed only from controller thread)
        self.desired_q = np.array([0.0] * self.n)

        # Admittance control parameters (used in policy thread)
        self.Mc = np.diag([20.0, 20.0, 20.0, 5, 5, 5])
        self.Dc = np.diag([200.0, 200.0, 200.0, 50.0, 50.0, 50.0])
        self.Kc = np.diag([500.0, 500.0, 500.0, 200.0, 200.0, 200.0])

        # Admittance states (used in policy thread only)
        self.x_admittance = np.zeros(6)
        self.dx_admittance = np.zeros(6)
        self.ddx_admittance = np.zeros(6)
        self.admittance_lock = threading.Lock()

        # Trajectory recording
        self.trajectory_data = {
            'time': [], 'desired_pos': [], 'actual_pos': [], 'modified_pos': [],
            'desired_rot': [], 'actual_rot': [], 'modified_rot': [],
            'force': [], 'admittance_displacement': [], 'joint_angles': []
        }
        
        # Store latest robot state for data recording
        self.latest_robot_state = None

        # Initialize trajectory functions
        self.pd_t, self.Rd_t, self.dpd_t, self.dRd_t, self.ddpd_t, self.ddRd_t = calculate_desired_pose_trajectory(self.task, self.duration)
        self.initial_q = self.get_initial_joint_config()
        
        # ROS2 publishers
        self.PublishMujocoSimClock = self.create_publisher(Clock, '/clock', 10)
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # Shared references for thread access
        self.model = None
        self.data = None
        self.robot_state = None
        self.ik_solver = None
        self.previous_q = None

    def get_initial_joint_config(self):
        """Get pre-computed initial joint configuration for each task"""
        initial_configs = {
            'sphere': np.array([-1.9953613783, 1.2212620712, -2.0331956982, 
                                    2.1375685376, 2.0386397961, 1.0805070204]),
            'regulation': np.array([-1.7671236707, 1.5031112517, -1.8753320848, 
                                    1.9429508990, 1.5703766827, 1.3875707847]),
            'circle': np.array([-1.6226059033, 1.4895827403, -1.8643317331, 
                                1.9405617237, 1.5719018365, 1.5311060553]),
            'line': np.array([-2.0984495472, 1.4330101841, -1.7939865607, 
                              1.9359154117, 1.5725287978, 1.0554072454])
        }
        return initial_configs.get(self.task, np.array([0.0, np.pi/2, 0.0, np.pi/2, 0.0, 0.0]))

    def policy_thread_worker(self):
        """Policy thread running at 25Hz - computes IK from Cartesian targets"""
        self.get_logger().info("Starting policy thread (25Hz)")
        
        while self.policy_running and self.simulation_running:
            loop_start = time.time()
            
            try:
                # Wait for new robot state data from main thread
                robot_state = self.policy_request_queue.get(timeout=0.1)
                
                if robot_state is None:
                    continue
                    
                if robot_state.timestamp <= self.duration and not self.paused:
                    # Compute desired pose from trajectory
                    pd_current = np.array(self.pd_t(robot_state.timestamp)).flatten()
                    Rd_current = np.array(self.Rd_t(robot_state.timestamp)).reshape(3, 3)
                    
                    # Update admittance dynamics using snapshot data
                    with self.admittance_lock:
                        # Current pose from snapshot
                        current_pose = np.eye(4)
                        current_pose[:3, :3] = robot_state.current_rotation
                        current_pose[:3, 3] = robot_state.current_position
                        
                        # Desired pose
                        desired_pose = np.eye(4) 
                        desired_pose[:3, :3] = Rd_current
                        desired_pose[:3, 3] = pd_current
                        
                        # Position error (translation)
                        pos_error = robot_state.current_position - pd_current
                        
                        # For rotation error, use a simplified approach
                        # We'll just use position error and force/torque directly
                        error_6d = np.concatenate([pos_error, np.zeros(3)])  # Simplified without rotation
                        
                        # Force/torque wrench
                        external_wrench = robot_state.force_torque
                        if len(external_wrench) != 6:
                            self.get_logger().warn(f"Unexpected force/torque shape: {external_wrench.shape}, resizing")
                            if len(external_wrench) > 6:
                                external_wrench = external_wrench[:6]
                            else:
                                external_wrench = np.pad(external_wrench, (0, 6-len(external_wrench)))
                        
                        wrench_6d = external_wrench
                        
                        # Admittance dynamics update
                        self.ddx_admittance = np.linalg.solve(self.Mc, 
                            wrench_6d - self.Dc @ self.dx_admittance - self.Kc @ self.x_admittance)
                        
                        self.dx_admittance += self.ddx_admittance * 0.04  # dt = 0.04s for 25Hz
                        self.x_admittance += self.dx_admittance * 0.04
                        
                        # Apply admittance to desired pose (only position for now)
                        pd_modified = pd_current + self.x_admittance[:3]
                        Rd_modified = Rd_current  # Keep original orientation for simplicity
                        
                        # Create target transform
                        Tep = np.eye(4)
                        Tep[:3, :3] = Rd_modified
                        Tep[:3, 3] = pd_modified
                    
                    # Solve IK using snapshot data 
                    # Note: We pass robot_state.q as the initial guess for IK
                    q_sol, success, iterations, error, jl_valid, solve_time = self.ik_solver.solve_ik_from_snapshot(
                        self.model, Tep, robot_state.q, robot_state.jacobian
                    )
                    
                    if success:
                        # Create policy output with timestamp
                        policy_output = PolicyOutput(
                            timestamp=robot_state.timestamp,
                            desired_q=q_sol.copy(),
                            pd_modified=pd_modified.copy(),
                            Rd_modified=Rd_modified.copy(),
                            x_admittance=self.x_admittance.copy(),
                            success=True
                        )
                        
                        # Send to interpolator
                        self.interpolator.add_sample(policy_output)
                        
                        if robot_state.timestamp % 2.0 < 0.04:
                            self.get_logger().info(
                                f"Policy: t={robot_state.timestamp:.2f}s, solve_time={solve_time*1000:.1f}ms"
                            )
                    else:
                        self.get_logger().warn(f"IK failed at t={robot_state.timestamp:.3f}s")
                        
            except Exception as e:
                self.get_logger().error(f"Policy thread error: {e}")
                
            # Maintain frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.04 - elapsed)  # 25Hz = 0.04s
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self.get_logger().info("Policy thread stopped")
        
    def controller_thread_worker(self):
        """Controller thread running at 125Hz - sends control commands"""
        self.get_logger().info("Starting controller thread (125Hz)")
        
        while self.controller_running and self.simulation_running:
            loop_start = time.time()
            
            try:
                with self.time_lock:
                    current_time = self.current_time
                
                if current_time <= self.duration and not self.paused:
                    # Get interpolated reference from policy outputs
                    interpolated_q, interpolated_data = self.interpolator.interpolate(current_time)
                    
                    if interpolated_q is not None:
                        # Apply filtering for smooth control
                        alpha = 0.8
                        self.desired_q = alpha * interpolated_q + (1 - alpha) * self.desired_q
                        
                        # Note: Control commands will be applied by main thread
                        # This thread only computes the desired values
                        
                        # Record data (at reduced frequency)
                        if current_time % 0.08 < 0.008 and interpolated_data is not None:  # ~12.5Hz logging
                            # Get current robot state snapshot for recording
                            try:
                                # Get current end-effector position from latest snapshot
                                if hasattr(self, 'latest_robot_state') and self.latest_robot_state is not None:
                                    # Get desired trajectory at current time
                                    pd_desired = self.pd_t(current_time).reshape(-1)
                                    Rd_desired = self.Rd_t(current_time)
                                    
                                    # Record trajectory data
                                    self.trajectory_data['time'].append(current_time)
                                    self.trajectory_data['desired_pos'].append(pd_desired.tolist())
                                    self.trajectory_data['actual_pos'].append(self.latest_robot_state.current_position.tolist())
                                    self.trajectory_data['modified_pos'].append(interpolated_data['pd_modified'].tolist())
                                    self.trajectory_data['desired_rot'].append(Rd_desired.flatten().tolist())
                                    self.trajectory_data['actual_rot'].append(self.latest_robot_state.current_rotation.flatten().tolist())
                                    self.trajectory_data['modified_rot'].append(interpolated_data['Rd_modified'].flatten().tolist())
                                    self.trajectory_data['force'].append(self.latest_robot_state.force_torque.tolist())
                                    self.trajectory_data['admittance_displacement'].append(interpolated_data['x_admittance'].tolist())
                                    self.trajectory_data['joint_angles'].append(self.desired_q.tolist())
                                    
                                    self.get_logger().debug(f"Controller: t={current_time:.3f}s, data recorded")
                                    
                            except Exception as e:
                                self.get_logger().error(f"Data recording error: {e}")
                    
            except Exception as e:
                self.get_logger().error(f"Controller thread error: {e}")
                
            # Maintain frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.008 - elapsed)  # 125Hz = 0.008s
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self.get_logger().info("Controller thread stopped")

    def update_admittance_dynamics(self, Fe, dt):
        """Update admittance dynamics"""
        if Fe.ndim > 1:
            Fe = Fe.flatten()
        
        Mc_inv = np.linalg.inv(self.Mc)
        self.ddx_admittance = Mc_inv @ (Fe - self.Dc @ self.dx_admittance - self.Kc @ self.x_admittance)
        
        self.dx_admittance += self.ddx_admittance * dt
        self.x_admittance += self.dx_admittance * dt
        
        # Apply limits
        max_translation = 0.1
        max_rotation = 0.2
        
        self.x_admittance[:3] = np.clip(self.x_admittance[:3], -max_translation, max_translation)
        self.x_admittance[3:] = np.clip(self.x_admittance[3:], -max_rotation, max_rotation)
        
        return self.x_admittance.copy()

    def apply_admittance_to_desired_pose(self, pd_desired, Rd_desired, x_admittance):
        """Apply admittance displacement to pose"""
        pd_modified = pd_desired + x_admittance[:3]
        
        rotation_displacement = x_admittance[3:]
        angle = np.linalg.norm(rotation_displacement)
        
        if angle > 1e-6:
            R_displacement = expm(hat_map(rotation_displacement))
            Rd_modified = R_displacement @ Rd_desired
        else:
            Rd_modified = Rd_desired.copy()
            
        return pd_modified, Rd_modified

    def MujocoSim(self):
        """Asynchronous simulation with policy and controller threads"""
        # Initialize MuJoCo
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.001  
        self.model.opt.iterations = 300    
        self.model.opt.tolerance = 1e-6  
        # self.model.opt.solver = mujoco.mjtSolver.mjSOL_PGS  # PGS is faster than Newton

        self.robot_state = RobotState(self.model, self.data, "jaka_end_effector", "jaka")
        self.ik_solver = IKArm(solver_type='QP', tol=1e-5, ilimit=10000)
        
        # Initialize state
        self.data.qpos[:self.n] = self.initial_q
        self.previous_q = self.initial_q.copy()
        self.desired_q = self.initial_q.copy()
        
        # Initialize interpolator
        initial_pd = np.array(self.pd_t(0.0)).flatten()
        initial_Rd = np.array(self.Rd_t(0.0)).reshape(3, 3)
        initial_policy_output = PolicyOutput(
            timestamp=0.0,
            desired_q=self.initial_q.copy(),
            pd_modified=initial_pd.copy(),
            Rd_modified=initial_Rd.copy(),
            x_admittance=np.zeros(6),
            success=True
        )
        self.interpolator.add_sample(initial_policy_output)
        
        with self.time_lock:
            self.current_time = 0.0
        
        # Start threads
        self.simulation_running = True
        self.policy_running = True
        self.controller_running = True
        
        self.policy_thread = threading.Thread(target=self.policy_thread_worker, daemon=True)
        self.controller_thread = threading.Thread(target=self.controller_thread_worker, daemon=True)
        
        self.policy_thread.start()
        self.controller_thread.start()
        
        self.get_logger().info(f"Starting simulation for {self.duration}s")
        self.get_logger().info(f"Policy: {self.policy_frequency}Hz, Controller: {self.controller_frequency}Hz")
        
        # Main simulation loop - non-blocking
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            last_policy_snapshot_time = 0.0
            policy_snapshot_period = 0.04  # 25Hz for policy thread
            
            while viewer.is_running() and self.current_time < self.duration:
                step_start = time.time()

                if not self.paused:
                    # Apply control commands from controller thread
                    if hasattr(self, 'desired_q') and self.desired_q is not None:
                        self.data.ctrl[:] = self.desired_q
                    
                    # Physics step (fast, non-blocking)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    # Update time
                    with self.time_lock:
                        self.current_time += self.model.opt.timestep
                        current_time_local = self.current_time
                    
                    # Send robot state snapshot to policy thread (at 25Hz)
                    if current_time_local - last_policy_snapshot_time >= policy_snapshot_period:
                        try:
                            # Create robot state snapshot (MuJoCo access only in main thread)
                            current_q = self.data.qpos[:self.n].copy()
                            mujoco.mj_forward(self.model, self.data)
                            
                            # Get end-effector pose
                            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ur10_ee")
                            current_position = self.data.site_xpos[site_id].copy()
                            current_rotation = self.data.site_xmat[site_id].reshape(3, 3).copy()
                            
                            # Get Jacobian
                            jacp = np.zeros((3, self.model.nv))
                            jacr = np.zeros((3, self.model.nv)) 
                            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
                            jacobian = np.vstack([jacp[:, :self.n], jacr[:, :self.n]])  # Shape: (6, n)
                            
                            # Get force/torque sensor data
                            if len(self.data.sensordata) >= 6:
                                # Use the first 6 values as force/torque (fx, fy, fz, tx, ty, tz)
                                force_torque = self.data.sensordata[:6].copy()
                            else:
                                # No sensor data available, use zeros
                                force_torque = np.zeros(6)
                            
                            # Create snapshot
                            robot_state = RobotStateSnapshot(
                                timestamp=current_time_local,
                                q=current_q,
                                current_position=current_position,
                                current_rotation=current_rotation,
                                jacobian=jacobian,
                                force_torque=force_torque
                            )
                            
                            # Send to policy thread
                            self.policy_request_queue.put_nowait(robot_state)
                            
                            # Update latest robot state for data recording
                            self.latest_robot_state = robot_state
                            
                            last_policy_snapshot_time = current_time_local
                            
                        except queue.Full:
                            # Queue full, skip this snapshot
                            pass
                        except Exception as e:
                            self.get_logger().error(f"Snapshot creation error: {e}")
                
                # Publish ROS messages
                joint_state_msg = JointState()
                joint_state_msg.position = [self.data.qpos[i] for i in range(self.n)]
                self.joint_state_publisher.publish(joint_state_msg)

                clock_msg = Clock()
                t_msg = Time()
                with self.time_lock:
                    current_time_local = self.current_time
                t_msg.sec = int(current_time_local)
                t_msg.nanosec = int((current_time_local - int(current_time_local)) * 1e9)
                clock_msg.clock = t_msg
                self.PublishMujocoSimClock.publish(clock_msg)

                # Maintain simulation frequency
                elapsed = time.time() - step_start
                sleep_time = max(0, self.model.opt.timestep - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                rclpy.spin_once(self, timeout_sec=0.0)
            
        # Clean shutdown
        self.get_logger().info("Simulation finished, stopping threads...")
        self.simulation_running = False
        self.policy_running = False
        self.controller_running = False
        
        if self.policy_thread and self.policy_thread.is_alive():
            self.policy_thread.join(timeout=1.0)
        if self.controller_thread and self.controller_thread.is_alive():
            self.controller_thread.join(timeout=1.0)
            
        self.get_logger().info("All threads stopped")
        self.save_trajectory_data()

    def save_trajectory_data(self):
        """Save trajectory data to files"""
        import pickle
        import csv
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'log')
        os.makedirs(data_dir, exist_ok=True)
        
        pickle_file = os.path.join(data_dir, f'trajectory_{self.task}.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.trajectory_data, f)
        
        csv_file = os.path.join(data_dir, f'trajectory_{self.task}.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['time', 'desired_x', 'desired_y', 'desired_z',
                     'actual_x', 'actual_y', 'actual_z',
                     'modified_x', 'modified_y', 'modified_z',
                     'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z',
                     'adm_disp_x', 'adm_disp_y', 'adm_disp_z', 'adm_disp_rx', 'adm_disp_ry', 'adm_disp_rz']
            header.extend([f'joint_{i+1}' for i in range(self.n)])
            writer.writerow(header)
            
            for i in range(len(self.trajectory_data['time'])):
                row = [self.trajectory_data['time'][i]]
                row.extend(self.trajectory_data['desired_pos'][i])
                row.extend(self.trajectory_data['actual_pos'][i])
                row.extend(self.trajectory_data['modified_pos'][i])
                row.extend(self.trajectory_data['force'][i])
                row.extend(self.trajectory_data['admittance_displacement'][i])
                row.extend(self.trajectory_data['joint_angles'][i])
                writer.writerow(row)
        
        self.get_logger().info(f"Async trajectory data saved to: {pickle_file}")

    def key_callback(self, keycode):
        """Key callback for simulation control"""
        if chr(keycode) == ' ':
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            with self.time_lock:
                current_time = self.current_time
            self.get_logger().info(f"Simulation {status} at t={current_time:.3f}s")


def main(args=None):
    rclpy.init(args=args)
    
    task = input("Enter task name ('regulation', 'circle', 'line', 'sphere'): ")
    
    if args and len(args) > 0:
        task = args[0]
    
    controller_node = MujocoNode(task=task)
    
    print(f"\nAsynchronous Control Architecture:")
    print(f"- Policy (IK): {controller_node.policy_frequency} Hz")
    print(f"- Controller: {controller_node.controller_frequency} Hz") 
    print("- Simulation: Continuous, non-blocking")
    print("\nControls: Space = Pause/Resume")

    try:
        controller_node.MujocoSim()
    except KeyboardInterrupt:
        pass
    finally:
        controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv[1:] if len(sys.argv) > 1 else None)