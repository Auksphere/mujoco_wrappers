#!/usr/bin/env python3
"""
Variable Admittance Control Expert Data Generator with AIRL PKL Export

This script generates expert trajectory data for the peg-in-hole task using
adaptive stiffness variable admittance control. It outputs AIRL-compatible 
PKL format for Adversarial Inverse Reinforcement Learning.

Usage:
    # Interactive mode
    python vac_expert_generator.py
    
    # Command line mode (specify task)
    python vac_expert_generator.py pih
    
    # Edit expert_pkl_name in MujocoSimulator.__init__ to control AIRL PKL export

AIRL Format for Variable Admittance Control:
    - Observations: 12D [e_t(3) + pd_t(3) + e_r(3) + rd_t(3)]
        - e_t = p - pd (tracking error), pd_t = desired position
        - e_r: rotation error (rotvec), rd_t: desired rotation (rotvec)
    - Actions: 7D [K1, K2, K3, K4, K5, K6, damping_ratio] impedance parameters
    
Control Law:
    - Variable Admittance: pd_new = pd + e_admittance
    - Admittance Dynamics: M*e'' + B*e' + K*e = F_external
    - Damping: B = damping_ratio * sqrt(K)
    
Configuration:
    - To enable AIRL PKL export: Set expert_pkl_name = 'your_filename.pkl' in MujocoSimulator.__init__
    - To disable AIRL PKL export: Set expert_pkl_name = None in MujocoSimulator.__init__
    
Author: GitHub Copilot
"""
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

import mujoco
import mujoco.viewer
import time
import numpy as np
import sys
import argparse
import os
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from misc_func import calculate_desired_pose_trajectory, vee_map, hat_map, adjoint_g_ed, adjoint_g_ed_dual, adjoint_g
from filter import ButterLowPass
from scipy.linalg import block_diag, expm
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp

# Add project root to Python path to allow importing from 'controllers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from hyperparams import get_vac_hyperparams
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
    distance_to_hole: float
    stiffness_norm: float
    transition_factor: float
    damping_ratio: float  # Add damping ratio to policy output
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
                    'x_admittance': sample.x_admittance.copy(),
                    'distance_to_hole': sample.distance_to_hole,
                    'stiffness_norm': sample.stiffness_norm,
                    'transition_factor': sample.transition_factor,
                    'damping_ratio': sample.damping_ratio
                }
                
            # Find interpolation interval
            times = [sample.timestamp for sample in self.buffer]
            
            if target_time <= times[0]:
                sample = self.buffer[0]
                return sample.desired_q.copy(), {
                    'pd_modified': sample.pd_modified.copy(),
                    'Rd_modified': sample.Rd_modified.copy(),
                    'x_admittance': sample.x_admittance.copy(),
                    'distance_to_hole': sample.distance_to_hole,
                    'stiffness_norm': sample.stiffness_norm,
                    'transition_factor': sample.transition_factor,
                    'damping_ratio': sample.damping_ratio
                }
            elif target_time >= times[-1]:
                sample = self.buffer[-1] 
                return sample.desired_q.copy(), {
                    'pd_modified': sample.pd_modified.copy(),
                    'Rd_modified': sample.Rd_modified.copy(),
                    'x_admittance': sample.x_admittance.copy(),
                    'distance_to_hole': sample.distance_to_hole,
                    'stiffness_norm': sample.stiffness_norm,
                    'transition_factor': sample.transition_factor,
                    'damping_ratio': sample.damping_ratio
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
                
                # Interpolate scalar values
                dist_interp = (1 - alpha) * sample1.distance_to_hole + alpha * sample2.distance_to_hole
                stiff_interp = (1 - alpha) * sample1.stiffness_norm + alpha * sample2.stiffness_norm
                trans_interp = (1 - alpha) * sample1.transition_factor + alpha * sample2.transition_factor
                damp_interp = (1 - alpha) * sample1.damping_ratio + alpha * sample2.damping_ratio
                
                # SLERP for rotation matrices
                R_interp = self._slerp_rotation(sample1.Rd_modified, sample2.Rd_modified, alpha)
                
                return interpolated_q, {
                    'pd_modified': pd_interp,
                    'Rd_modified': R_interp,
                    'x_admittance': x_interp,
                    'distance_to_hole': dist_interp,
                    'stiffness_norm': stiff_interp,
                    'transition_factor': trans_interp,
                    'damping_ratio': damp_interp
                }

    def get_buffer_length(self) -> int:
        """Return the current number of samples in the interpolator buffer (thread-safe)."""
        with self.lock:
            return len(self.buffer)
                
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

class MujocoSimulator:
    def __init__(self, task='pih', trajectory_index=0, index_offset=0, mode=None):
        self.n = 6
        self.xml_file = 'models/jaka_zu12/jaka_pih_case0.xml'
        self.task = task
        self.trajectory_index = trajectory_index
        self.vac_cfg = get_vac_hyperparams()
        
        # Generate unique PKL filename for each trajectory
        self.expert_pkl_name = f'expert_{task}_{trajectory_index+index_offset}.pkl'
        

        self.duration = float(self.vac_cfg['duration_sec'])
        self.success_xy_threshold = float(self.vac_cfg['success_xy_threshold'])
        self.success_z_threshold = float(self.vac_cfg['success_z_threshold'])
        self.insertion_success = False
        self.termination_reason = "time_limit"
            
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

        self.desired_q = np.array([0.0] * self.n)

        # Admittance control parameters (used in policy thread)
        self.d_far = float(self.vac_cfg['d_far'])
        self.d_near = float(self.vac_cfg['d_near'])
        self.d = self.d_far 
        
        self.Mc = self.vac_cfg['Mc'].copy()
        
        # Adaptive Kc parameters
        self.Kc_far = self.vac_cfg['Kc_far'].copy()
        self.Kc_near = self.vac_cfg['Kc_near'].copy()
        self.Kc = self.Kc_far.copy()  
        self.distance_threshold = float(self.vac_cfg['distance_threshold'])
        self.hole_position = self.vac_cfg['hole_position'].copy()
        
        self.Dc = self.Kc * self.d

        self.x_admittance = np.zeros(6)
        self.dx_admittance = np.zeros(6)
        self.ddx_admittance = np.zeros(6)
        self.admittance_lock = threading.Lock()

        # Trajectory recording
        self.trajectory_data = {
            'time': [], 'desired_pos': [], 'actual_pos': [], 'modified_pos': [],
            'desired_rot': [], 'actual_rot': [], 'modified_rot': [],
            'force': [], 'admittance_displacement': [], 'joint_angles': [],
            'distance_to_hole': [], 'stiffness_norm': [], 'transition_factor': [],
            'damping_ratio': [],
            'insertion_success': [],
            'K1': [], 'K2': [], 'K3': [], 'K4': [], 'K5': [], 'K6': []
        }
        
        self.latest_robot_state = None

        from misc_func import get_initial_joint_config
        self.initial_q, initial_pose = get_initial_joint_config(self.task, self.xml_file, IKArm, mode)

        self.pd_t, self.Rd_t, self.dpd_t, self.dRd_t, self.ddpd_t, self.ddRd_t = calculate_desired_pose_trajectory(self.task, self.duration, initial_config=initial_pose)
        
        self.model = None
        self.data = None
        self.robot_state = None
        self.ik_solver = None
        self.previous_q = None

        # Wrench preprocessing (bias compensation + low-pass + median)
        self.wrench_filter = None
        self.wrench_cutoff_hz = 15.0
        self.wrench_filter_order = 3
        self.wrench_use_median = True
        self.wrench_median_window = 5
        self.wrench_median_buffer = deque(maxlen=self.wrench_median_window)
        self.wrench_bias = np.zeros(6, dtype=np.float64)
        self.wrench_bias_steps = int(self.vac_cfg.get('wrench_bias_steps', 500))
        self.wrench_bias_count = 0
        self.use_wrench_bias_prior = bool(self.vac_cfg.get('use_wrench_bias_prior', True))
        self.wrench_bias_prior = np.asarray(
            self.vac_cfg.get('wrench_bias_prior', np.zeros(6, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(6)
        self.wrench_preprocessed = np.zeros(6, dtype=np.float64)

    def _read_wrench_from_model_sensors(self):
        """Read 6D wrench [fx, fy, fz, mx, my, mz] from MuJoCo sensors."""
        def read_sensor_vec(sensor_name, expected_dim=3):
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sensor_id < 0:
                raise ValueError(f"Sensor not found: {sensor_name}")
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            vec = np.copy(self.data.sensordata[adr:adr + dim]).reshape(-1)
            if vec.shape[0] >= expected_dim:
                return vec[:expected_dim]
            return np.pad(vec, (0, expected_dim - vec.shape[0]))

        try:
            # Preferred: legacy 3D vector sensors (single force + single torque)
            force = read_sensor_vec("jaka_force_sensor", expected_dim=3)
            torque = read_sensor_vec("jaka_torque_sensor", expected_dim=3)
        except Exception:
            # Fallback to named force/torque sensors.
            force = read_sensor_vec("jaka_force_sensor_fx", expected_dim=3)
            torque = read_sensor_vec("jaka_torque_sensor_mx", expected_dim=3)

        return np.concatenate([force, torque])

    def _preprocess_wrench_1khz(self):
        raw_wrench = self._read_wrench_from_model_sensors()
        raw_wrench = np.nan_to_num(raw_wrench, nan=0.0, posinf=0.0, neginf=0.0)

        self.wrench_bias_count += 1
        if self.wrench_bias_count <= self.wrench_bias_steps:
            alpha = 1.0 / float(self.wrench_bias_count)
            self.wrench_bias = (1.0 - alpha) * self.wrench_bias + alpha * raw_wrench

        bias_removed = raw_wrench - self.wrench_bias

        if self.wrench_filter is not None:
            filtered = self.wrench_filter(bias_removed.reshape(1, -1))[0]
        else:
            filtered = bias_removed

        if self.wrench_use_median:
            self.wrench_median_buffer.append(filtered.copy())
            if len(self.wrench_median_buffer) >= 3:
                filtered = np.median(np.stack(self.wrench_median_buffer, axis=0), axis=0)

        self.wrench_preprocessed = np.asarray(filtered, dtype=np.float64).reshape(6)
        return self.wrench_preprocessed.copy()

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
                        # error_6d = np.concatenate([pos_error, np.zeros(3)])  # Simplified without rotation
                        
                        # Force/torque wrench
                        external_wrench = robot_state.force_torque
                        if len(external_wrench) != 6:
                            self.get_logger().warn(f"Unexpected force/torque shape: {external_wrench.shape}, resizing")
                            if len(external_wrench) > 6:
                                external_wrench = external_wrench[:6]
                            else:
                                external_wrench = np.pad(external_wrench, (0, 6-len(external_wrench)))
                        
                        wrench_6d = external_wrench
                        
                        # Update adaptive stiffness based on peg-hole distance
                        distance_to_hole, transition_factor = self.update_adaptive_stiffness(robot_state.current_position)
                        
                        # Admittance dynamics update
                        self.ddx_admittance = np.linalg.solve(self.Mc, 
                            wrench_6d - self.Dc @ self.dx_admittance - self.Kc @ self.x_admittance)
                        
                        self.dx_admittance += self.ddx_admittance * 0.04  # dt = 0.04s for 25Hz
                        self.x_admittance += self.dx_admittance * 0.04
                        
                        # Apply 6D admittance displacement to desired pose.
                        pd_modified, Rd_modified = self.apply_admittance_to_desired_pose(
                            pd_current,
                            Rd_current,
                            self.x_admittance,
                        )
                        
                        # Create target transform
                        Tep = np.eye(4)
                        Tep[:3, :3] = Rd_modified
                        Tep[:3, 3] = pd_modified
                    
                    # Solve IK using snapshot data 
                    q_sol, success, iterations, error, jl_valid, solve_time = self.ik_solver.solve_ik_from_snapshot(
                        self.model, Tep, robot_state.q, robot_state.jacobian
                    )
                    
                    if success:
                        # Create policy output with timestamp
                        kc_norm = np.linalg.norm(np.diag(self.Kc[:3, :3]))
                        policy_output = PolicyOutput(
                            timestamp=robot_state.timestamp,
                            desired_q=q_sol.copy(),
                            pd_modified=pd_modified.copy(),
                            Rd_modified=Rd_modified.copy(),
                            x_admittance=self.x_admittance.copy(),
                            distance_to_hole=distance_to_hole,
                            stiffness_norm=kc_norm,
                            transition_factor=transition_factor,
                            damping_ratio=self.d,  # Include current adaptive damping
                            success=True
                        )
                        
                        self.interpolator.add_sample(policy_output)
                        
                        if robot_state.timestamp % 2.0 < 0.04:
                            # Get current stiffness for logging
                            kc_norm = np.linalg.norm(np.diag(self.Kc[:3, :3]))
                            self.get_logger().info(
                                f"Policy: t={robot_state.timestamp:.2f}s, solve_time={solve_time*1000:.1f}ms, "
                                f"dist_to_hole={distance_to_hole*100:.1f}cm, Kc_norm={kc_norm:.0f}, "
                                f"transition={transition_factor:.2f}"
                            )
                    else:
                        self.get_logger().warn(f"IK failed at t={robot_state.timestamp:.3f}s")
                        
            except Exception as e:
                self.get_logger().error(f"Policy thread error: {e}")
                
            # Maintain frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.04 - elapsed) 
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

                        
                        # Record data (at reduced frequency)
                        if current_time % 0.08 < 0.008 and interpolated_data is not None:
                            try:
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

                                    try:
                                        buf_len = self.interpolator.get_buffer_length()
                                    except Exception:
                                        buf_len = 0
                                    if buf_len >= 1 and interpolated_data is not None and 'distance_to_hole' in interpolated_data:
                                        # Trust interpolated value when there are enough samples
                                        dist_to_hole = float(interpolated_data['distance_to_hole'])
                                    else:
                                        try:
                                            dist_to_hole = float(np.linalg.norm(self.latest_robot_state.current_position - self.hole_position))
                                        except Exception:
                                            dist_to_hole = interpolated_data.get('distance_to_hole') if interpolated_data is not None else 0.0

                                    self.trajectory_data['distance_to_hole'].append(dist_to_hole)
                                    self.trajectory_data['stiffness_norm'].append(interpolated_data['stiffness_norm'])
                                    self.trajectory_data['transition_factor'].append(interpolated_data['transition_factor'])
                                    self.trajectory_data['damping_ratio'].append(interpolated_data['damping_ratio'])
                                    self.trajectory_data['insertion_success'].append(bool(self.insertion_success))

                                    # Compute and record per-axis stiffness (K1-K6). If interpolated transition factor
                                    # is available, reconstruct K diag from near/far presets; otherwise use current Kc.
                                    try:
                                        tf = interpolated_data.get('transition_factor', None)
                                    except Exception:
                                        tf = None

                                    if tf is not None:
                                        Kc_far_diag = np.diag(self.Kc_far)
                                        Kc_near_diag = np.diag(self.Kc_near)
                                        K_diag = tf * Kc_far_diag + (1.0 - tf) * Kc_near_diag
                                    else:
                                        K_diag = np.diag(self.Kc)

                                    for k_idx in range(6):
                                        self.trajectory_data[f'K{k_idx+1}'].append(float(K_diag[k_idx]))
                                    
                                    self.get_logger().debug(f"Controller: t={current_time:.3f}s, data recorded")
                                    
                            except Exception as e:
                                self.get_logger().error(f"Data recording error: {e}")
                    
            except Exception as e:
                self.get_logger().error(f"Controller thread error: {e}")
                
            # Maintain frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.008 - elapsed) 
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self.get_logger().info("Controller thread stopped")
    
    def get_logger(self):
        """Simple logger for compatibility"""
        class SimpleLogger:
            def info(self, msg):
                print(f"[INFO] {msg}")
            def warn(self, msg):
                print(f"[WARN] {msg}")
            def error(self, msg):
                print(f"[ERROR] {msg}")
            def debug(self, msg):
                pass  # Skip debug messages
        return SimpleLogger()

    def update_adaptive_stiffness(self, peg_position):
        """Update Kc and d based on distance between peg and hole"""
        # Calculate distance between peg and hole
        distance_to_hole = np.linalg.norm(peg_position - self.hole_position)
        transition_sharpness = 50.0  # Controls how sharp the transition is
        transition_factor = 1.0 / (1.0 + np.exp(-transition_sharpness * (distance_to_hole - self.distance_threshold)))
        
        # Interpolate between near and far stiffness
        self.Kc = transition_factor * self.Kc_far + (1.0 - transition_factor) * self.Kc_near
        
        # Interpolate between near and far damping (opposite trend to stiffness)
        self.d = transition_factor * self.d_far + (1.0 - transition_factor) * self.d_near
        
        # Update damping coefficient accordingly
        self.Dc = self.Kc * self.d
        
        return distance_to_hole, transition_factor

    def _check_insertion_success(self):
        """Insertion success: aligned in XY and close in Z to hole center."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "jaka_end_effector")
        ee_pos = self.data.site_xpos[site_id].copy()
        xy_error = np.linalg.norm(ee_pos[:2] - self.hole_position[:2])
        z_error = abs(ee_pos[2] - self.hole_position[2])
        return (xy_error < self.success_xy_threshold) and (z_error < self.success_z_threshold)

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
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.001  
        self.model.opt.iterations = 300    
        self.model.opt.tolerance = 1e-6  

        fs = 1.0 / self.model.opt.timestep
        self.wrench_filter = ButterLowPass(self.wrench_cutoff_hz, fs, order=self.wrench_filter_order)
        self.wrench_filter.reset_state()
        self.wrench_median_buffer.clear()
        if self.use_wrench_bias_prior:
            self.wrench_bias[:] = self.wrench_bias_prior
            self.wrench_bias_count = self.wrench_bias_steps
        else:
            self.wrench_bias[:] = 0.0
            self.wrench_bias_count = 0
        self.wrench_preprocessed[:] = 0.0

        self.robot_state = RobotState(self.model, self.data, "jaka_end_effector", "jaka")
        self.ik_solver = IKArm(solver_type='QP', tol=1e-5, ilimit=10000)
        
        # Initialize state
        self.data.qpos[:self.n] = self.initial_q
        self.previous_q = self.initial_q.copy()
        self.desired_q = self.initial_q.copy()
        
        # Initialize interpolator
        initial_pd = np.array(self.pd_t(0.0)).flatten()
        initial_Rd = np.array(self.Rd_t(0.0)).reshape(3, 3)
        initial_distance = np.linalg.norm(initial_pd - self.hole_position)
        initial_policy_output = PolicyOutput(
            timestamp=0.0,
            desired_q=self.initial_q.copy(),
            pd_modified=initial_pd.copy(),
            Rd_modified=initial_Rd.copy(),
            x_admittance=np.zeros(6),
            distance_to_hole=initial_distance,
            stiffness_norm=np.linalg.norm(np.diag(self.Kc_far[:3, :3])),
            transition_factor=1.0,  
            damping_ratio=self.d_far, 
            success=True
        )
        self.interpolator.add_sample(initial_policy_output)
        
        with self.time_lock:
            self.current_time = 0.0

        self.simulation_running = True
        self.policy_running = True
        self.controller_running = True
        
        self.policy_thread = threading.Thread(target=self.policy_thread_worker, daemon=True)
        self.controller_thread = threading.Thread(target=self.controller_thread_worker, daemon=True)
        
        self.policy_thread.start()
        self.controller_thread.start()
        
        self.get_logger().info(f"Starting simulation for {self.duration}s")
        self.get_logger().info(f"Policy: {self.policy_frequency}Hz, Controller: {self.controller_frequency}Hz")
        
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            last_policy_snapshot_time = 0.0
            policy_snapshot_period = 0.04
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "jaka_end_effector")
            
            while viewer.is_running() and self.current_time < self.duration and not self.insertion_success:
                step_start = time.time()

                if not self.paused:
                    if hasattr(self, 'desired_q') and self.desired_q is not None:
                        self.data.ctrl[:] = self.desired_q
                    
                    # Physics step (fast, non-blocking)
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                    if self._check_insertion_success():
                        self.insertion_success = True
                        self.termination_reason = "success"
                        self.get_logger().info(
                            f"Insertion success at t={self.current_time:.3f}s, stopping simulation early"
                        )
                        break
                    
                    # Update time
                    with self.time_lock:
                        self.current_time += self.model.opt.timestep
                        current_time_local = self.current_time

                    self._preprocess_wrench_1khz()
                    
                    # Send robot state snapshot to policy thread (at 25Hz)
                    if current_time_local - last_policy_snapshot_time >= policy_snapshot_period:
                        try:
                            # Create robot state snapshot (MuJoCo access only in main thread)
                            current_q = self.data.qpos[:self.n].copy()
                            mujoco.mj_forward(self.model, self.data)
                            
                            # Get end-effector pose
                            current_position = self.data.site_xpos[site_id].copy()
                            current_rotation = self.data.site_xmat[site_id].reshape(3, 3).copy()
                            
                            # Get Jacobian
                            jacp = np.zeros((3, self.model.nv))
                            jacr = np.zeros((3, self.model.nv)) 
                            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
                            jacobian = np.vstack([jacp[:, :self.n], jacr[:, :self.n]])  # Shape: (6, n)
                            
                            # Use preprocessed wrench (bias compensated and filtered)
                            force_torque = self.wrench_preprocessed.copy()
                            
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
                            pass
                        except Exception as e:
                            self.get_logger().error(f"Snapshot creation error: {e}")
                
                # Maintain simulation frequency
                elapsed = time.time() - step_start
                sleep_time = max(0, self.model.opt.timestep - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
        if not self.insertion_success and self.current_time >= self.duration:
            self.termination_reason = "time_limit"

        self.get_logger().info(
            f"Simulation finished (reason={self.termination_reason}, success={self.insertion_success}), stopping threads..."
        )
        self.simulation_running = False
        self.policy_running = False
        self.controller_running = False
        
        if self.policy_thread and self.policy_thread.is_alive():
            self.policy_thread.join(timeout=1.0)
        if self.controller_thread and self.controller_thread.is_alive():
            self.controller_thread.join(timeout=1.0)
            
        self.get_logger().info("All threads stopped")
        self.save_trajectory_data(expert_pkl_name=getattr(self, 'expert_pkl_name', None))

    def save_trajectory_data(self, expert_pkl_name=None):
        """Save trajectory data to files"""
        import pickle
        import csv
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'log')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate unique CSV filename with trajectory index
        csv_file = os.path.join(log_dir, f'trajectory_{self.task}_expert.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['time', 'desired_x', 'desired_y', 'desired_z',
                     'actual_x', 'actual_y', 'actual_z',
                     'modified_x', 'modified_y', 'modified_z',
                     'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z',
                     'adm_disp_x', 'adm_disp_y', 'adm_disp_z', 'adm_disp_rx', 'adm_disp_ry', 'adm_disp_rz',
                     'distance_to_hole', 'stiffness_norm', 'transition_factor', 'damping_ratio']
            # Add per-axis stiffness columns
            header.extend([f'K{i+1}' for i in range(6)])
            header.extend([f'joint_{i+1}' for i in range(self.n)])
            writer.writerow(header)
            
            for i in range(len(self.trajectory_data['time'])):
                row = [self.trajectory_data['time'][i]]
                row.extend(self.trajectory_data['desired_pos'][i])
                row.extend(self.trajectory_data['actual_pos'][i])
                row.extend(self.trajectory_data['modified_pos'][i])
                row.extend(self.trajectory_data['force'][i])
                row.extend(self.trajectory_data['admittance_displacement'][i])
                row.append(self.trajectory_data['distance_to_hole'][i])
                row.append(self.trajectory_data['stiffness_norm'][i])
                row.append(self.trajectory_data['transition_factor'][i])
                row.append(self.trajectory_data['damping_ratio'][i])  # Add damping ratio to CSV
                # Append per-axis stiffness values K1..K6
                for k_idx in range(6):
                    row.append(self.trajectory_data.get(f'K{k_idx+1}', [0]*len(self.trajectory_data['time']))[i])
                row.extend(self.trajectory_data['joint_angles'][i])
                writer.writerow(row)
        
        self.get_logger().info(f"CSV trajectory data saved to: {csv_file}")

        if expert_pkl_name:
            expert_data = self._convert_to_irl_dataset()
            expert_pkl_file = os.path.join(data_dir, expert_pkl_name)
            with open(expert_pkl_file, 'wb') as f:
                pickle.dump(expert_data, f)
            self.get_logger().info(f"IRL expert dataset saved to: {expert_pkl_file}")
        

    def _convert_to_irl_dataset(self):
        """Convert trajectory data to Variable Admittance Control IRL dataset format
        
        Returns:
            dict: IRL dataset with observations and actions
                observations (list): 12D observations [e_t(3) + pd_t(3) + e_r(3) + rd_t(3)]
                actions (list): 7D actions [K1, K2, K3, K4, K5, K6, damping_ratio]
        """
        observations = []
        actions = []
        
        print(f"Converting {len(self.trajectory_data['time'])} trajectory points to Variable Admittance Control IRL format...")
        
        for i, t in enumerate(self.trajectory_data['time']):
            # Get current and desired positions
            actual_pos = np.array(self.trajectory_data['actual_pos'][i])     # p(t): Current position
            desired_pos = np.array(self.trajectory_data['desired_pos'][i])   # pd(t): Desired position

            # Get current and desired rotations
            actual_rot = np.array(self.trajectory_data['actual_rot'][i]).reshape(3, 3)    # R(t)
            desired_rot = np.array(self.trajectory_data['desired_rot'][i]).reshape(3, 3)  # Rd(t)
            
            # =================== VARIABLE ADMITTANCE CONTROL STATE ===================
            # Tracking error: e_t = p(t) - pd(t) (actual - desired)
            tracking_error = actual_pos - desired_pos  # e_t (3D)

            # Desired position: pd_t (3D)
            desired_position = desired_pos.copy()  # pd_t (3D)

            # Rotation error: e_r = Log(R * Rd^T) as rotation vector (rotvec)
            R_err = actual_rot @ desired_rot.T
            rot_error = self._rotation_matrix_to_axis_angle(R_err)  # e_r (3D)

            # Desired rotation as rotvec for state input
            desired_rotvec = self._rotation_matrix_to_axis_angle(desired_rot)  # rd_t (3D)

            # =================== 12D OBSERVATION ===================
            observation = np.concatenate([
                tracking_error,     # e_t (3D): actual_pos - desired_pos
                desired_position,   # pd_t (3D): desired_position
                rot_error,          # e_r (3D)
                desired_rotvec      # rd_t (3D)
            ])
            
            # =================== 7D ACTION ===================
            # Extract expert impedance parameters.
            # Prefer using the recorded per-axis stiffness values K1..K6 that come directly
            # from the controller (this avoids any hard-coded ranges here).
            transition_factor = float(self.trajectory_data['transition_factor'][i])

            has_recorded_k = all(f'K{k+1}' in self.trajectory_data for k in range(6))
            if has_recorded_k and len(self.trajectory_data['K1']) > i:
                try:
                    K1 = float(self.trajectory_data['K1'][i])
                    K2 = float(self.trajectory_data['K2'][i])
                    K3 = float(self.trajectory_data['K3'][i])
                    K4 = float(self.trajectory_data['K4'][i])
                    K5 = float(self.trajectory_data['K5'][i])
                    K6 = float(self.trajectory_data['K6'][i])
                except Exception:
                    has_recorded_k = False

            if not has_recorded_k:
                # Fallback: derive from simulator parameters (non-hardcoded).
                # This assumes the expert stiffness is interpolated between Kc_far and Kc_near.
                Kc_far_diag = np.diag(self.Kc_far).astype(float)
                Kc_near_diag = np.diag(self.Kc_near).astype(float)
                expert_stiffness = transition_factor * Kc_far_diag + (1.0 - transition_factor) * Kc_near_diag
                K1, K2, K3, K4, K5, K6 = [float(x) for x in expert_stiffness]

            # Get the actual adaptive damping ratio used by the expert at this time point
            expert_damping = float(self.trajectory_data['damping_ratio'][i])
            
            # 7D action: [K1, K2, K3, K4, K5, K6, damping_ratio]
            action = np.array([K1, K2, K3, K4, K5, K6, expert_damping])
            
            observations.append(observation.tolist())
            actions.append(action.tolist())
        
        # Add debug information
        obs_array = np.array(observations)
        act_array = np.array(actions)

        print(f"Variable Admittance Control Dataset Statistics:")
        print(f"Observation statistics:")
        print(f"  Shape: {obs_array.shape}")
        print(f"  Tracking error e_t range: [{obs_array[:, :3].min():.4f}, {obs_array[:, :3].max():.4f}]")
        print(f"  Desired position pd_t range: [{obs_array[:, 3:6].min():.4f}, {obs_array[:, 3:6].max():.4f}]")
        print(f"  Rotation error e_r (rotvec) range: [{obs_array[:, 6:9].min():.4f}, {obs_array[:, 6:9].max():.4f}]")
        print(f"  Desired rotation rd_t (rotvec) range: [{obs_array[:, 9:12].min():.4f}, {obs_array[:, 9:12].max():.4f}]")
        
        print(f"Action statistics:")
        print(f"  Shape: {act_array.shape}")
        print(f"  K1-K3 (position) range: [{act_array[:, :3].min():.1f}, {act_array[:, :3].max():.1f}]")
        print(f"  K4-K6 (rotation) range: [{act_array[:, 3:6].min():.1f}, {act_array[:, 3:6].max():.1f}]")
        print(f"  Damping ratio range: [{act_array[:, 6].min():.3f}, {act_array[:, 6].max():.3f}]")
        
        return {
            'observations': observations,  # 12D: [e_t(3) + pd_t(3) + e_r(3) + rd_t(3)]
            'actions': actions,           # 7D: [K1, K2, K3, K4, K5, K6, d]
            'metadata': {
                'observation_dim': 12,
                'action_dim': 7,
                'trajectory_length': len(observations),
                'task': self.task,
                'insertion_success': bool(self.insertion_success),
                'termination_reason': self.termination_reason,
                'max_attempt_time_sec': float(self.duration),
                'observation_description': 'Variable Admittance Control: [e_t(3), pd_t(3), e_r(3), rd_t(3)] where e_t=p-pd, e_r=Log(R*Rd^T) (rotvec), rd_t=rotvec(Rd)',
                'action_description': 'impedance_parameters: [K1, K2, K3, K4, K5, K6, damping_ratio]',
                'error_convention': 'tracking_error: actual - desired (positive means overshoot)',
                'control_law': 'Variable Admittance: pd_new = pd + e_admittance'
            }
        }
    
    def _rotation_matrix_to_axis_angle(self, R):
        """Convert rotation matrix to axis-angle representation"""
        try:
            from scipy.spatial.transform import Rotation as ScipyRotation
            r = ScipyRotation.from_matrix(R)
            return r.as_rotvec()
        except:
            # Fallback implementation
            angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
            if np.sin(angle) == 0:
                return np.zeros(3)
            axis = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(angle))
            return axis * angle
    
    def _rotation_matrix_derivative_to_angular_velocity(self, R, dR):
        """Convert rotation matrix derivative to angular velocity"""
        # Angular velocity: ω = 2 * trace(dR * R^T) where the result is skew-symmetric
        # Extract the vector from the skew-symmetric matrix
        omega_hat = dR @ R.T
        return np.array([omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]])
        
    def key_callback(self, keycode):
        """Key callback for simulation control"""
        if chr(keycode) == ' ':
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            with self.time_lock:
                current_time = self.current_time
            self.get_logger().info(f"Simulation {status} at t={current_time:.3f}s")
            
            # print when paused
            if self.paused and hasattr(self, 'data') and self.data is not None:
                current_joint_positions = self.data.qpos[:self.n].copy()
                self.get_logger().info(f"Current joint positions: {current_joint_positions}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Variable Admittance Control Expert Data Generator")

    parser.add_argument("--mode", type=str, default=None, help="Initial config mode: 'test' for fixed seed, otherwise random")

    parsed = parser.parse_args(args if args is not None else [])

    task = "pih"
    # task = "regulation"
    
    index_offset = 25
    mode = parsed.mode
    if mode == 'test':
        num_trajectories = 1
    else:
        num_trajectories = 1

    print(f"\n=== Generating {num_trajectories} trajectory(s) for task '{task}' (mode={mode}) ===")
    
    for trajectory_idx in range(num_trajectories):
        print(f"\n--- Trajectory {trajectory_idx + 1}/{num_trajectories} ---")
        
        # Create simulator with unique PKL filename for each trajectory
        simulator = MujocoSimulator(task=task, trajectory_index=trajectory_idx, index_offset=index_offset, mode=mode)
        
        print(f"- Policy (IK): {simulator.policy_frequency} Hz")
        print(f"- Controller: {simulator.controller_frequency} Hz") 
        print(f"- Task: {task}")
        print(f"- Trajectory Index: {trajectory_idx + 1}")
        
        if simulator.expert_pkl_name:
            print(f"- IRL PKL Export: {simulator.expert_pkl_name}")
        else:
            print("- IRL PKL Export: Disabled")
        
        print("- Press Space to Pause/Resume during simulation")
        
        try:
            simulator.MujocoSim()
            print(f"✓ Trajectory {trajectory_idx + 1} completed successfully")
        except KeyboardInterrupt:
            print(f"✗ Trajectory {trajectory_idx + 1} interrupted by user")
            break
        except Exception as e:
            print(f"✗ Trajectory {trajectory_idx + 1} failed: {e}")
            continue


if __name__ == '__main__':
    import sys
    main(sys.argv[1:] if len(sys.argv) > 1 else None)