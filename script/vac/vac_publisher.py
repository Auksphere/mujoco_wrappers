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

import mujoco
import mujoco.viewer
import time
import numpy as np
import sys
import argparse
import os
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from misc_func import calculate_desired_pose_trajectory, vee_map, hat_map, adjoint_g_ed, adjoint_g_ed_dual, adjoint_g
from filter import ButterLowPass
from scipy.linalg import block_diag, expm
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp

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

class VariableImpedancePolicy(nn.Module):
    """
    Simplified and optimized policy network for impedance learning
    Input: [e_t, pd_t] (6D) - tracking error and desired position (normalized to [-1,1])
    Output: [K1, K2, K3, K4, K5, K6, d_t] (7D) - impedance parameters (normalized to [0,1])
    
    Key improvements:
    1. Simpler dual-encoder architecture
    2. Focused feature processing 
    3. Better parameter initialization
    4. Enhanced stability mechanisms
    """
    def __init__(self, state_dim=6, action_dim=7, hidden_dim=128):
        super(VariableImpedancePolicy, self).__init__()
        
        # Simplified dual encoders 
        self.error_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.position_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Simplified shared processing
        self.shared_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Focused stiffness branch
        self.stiffness_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 12)  # 6 * 2 for mean and log_std
        )
        
        # Enhanced damping branch with better exploration
        self.damping_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)  # mean and log_std for damping
        )
        
        # Improved initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Enhanced weight initialization for better learning dynamics"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with custom gain
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Special initialization for output layers
        with torch.no_grad():
            # Stiffness branch - encourage moderate initial values
            self.stiffness_branch[-1].weight.data *= 0.5
            self.stiffness_branch[-1].bias.data[:6] = 0.3   # mean
            self.stiffness_branch[-1].bias.data[6:] = -0.7  # log_std
            
            # Damping branch - better exploration initialization
            self.damping_branch[-1].weight.data *= 0.7
            self.damping_branch[-1].bias.data[0] = 0.2      # mean
            self.damping_branch[-1].bias.data[1] = -0.5     # log_std
    
    def forward(self, state):
        """Optimized forward pass with stability improvements"""
        # Set to evaluation mode for inference
        self.eval()
        
        # Input validation and cleanup
        if torch.isnan(state).any():
            state = torch.where(torch.isnan(state), torch.zeros_like(state), state)
        
        # Split and encode input
        tracking_error = state[:, :3]
        desired_position = state[:, 3:]
        
        error_features = self.error_encoder(tracking_error)
        position_features = self.position_encoder(desired_position)
        
        # Combine features
        combined_features = torch.cat([error_features, position_features], dim=-1)
        shared_features = self.shared_net(combined_features)
        
        # Generate outputs
        stiffness_output = self.stiffness_branch(shared_features)
        damping_output = self.damping_branch(shared_features)
        
        # Parse parameters
        stiffness_mean = stiffness_output[:, :6]
        stiffness_log_std = stiffness_output[:, 6:]
        damping_mean = damping_output[:, 0:1]
        damping_log_std = damping_output[:, 1:2]
        
        # Combine and clamp for stability
        mean = torch.cat([stiffness_mean, damping_mean], dim=-1)
        log_std = torch.cat([stiffness_log_std, damping_log_std], dim=-1)
        
        # Conservative clamping for numerical stability
        mean = torch.clamp(mean, min=-2.5, max=2.5)
        log_std = torch.clamp(log_std, min=-1.0, max=0.5)
        
        # Final NaN check
        if torch.isnan(mean).any() or torch.isnan(log_std).any():
            print("ERROR: NaN detected in policy output, using safe values")
            mean = torch.zeros_like(mean)
            log_std = torch.ones_like(log_std) * (-1.0)
        
        return mean, log_std
    
    def get_deterministic_action(self, state):
        """Get deterministic action (mean) for evaluation"""
        mean, _ = self.forward(state)
        return torch.sigmoid(mean)

class AIRLPolicyManager:
    """Manager for AIRL-trained policy inference"""
    
    def __init__(self, policy_path="script/models/airl/policy.pt", expert_data_path="data/expert_demonstration.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load policy network with matching architecture (hidden_dim=128)
        self.policy = VariableImpedancePolicy(hidden_dim=128).to(self.device)
        try:
            self.policy.load_state_dict(torch.load(policy_path, map_location=self.device))
            self.policy.eval()
            print(f"AIRL policy loaded from: {policy_path}")
        except Exception as e:
            print(f"Failed to load policy from {policy_path}: {e}")
            raise
        
        # Load expert data statistics for normalization
        self.load_data_statistics(expert_data_path)
        
    def load_data_statistics(self, expert_data_path):
        """Load data statistics for state/action normalization"""
        try:
            with open(expert_data_path, 'rb') as f:
                expert_data = pickle.load(f)
            
            observations = np.array(expert_data['observations'])
            actions = np.array(expert_data['actions'])
            
            # Calculate normalization statistics
            self.obs_stats = {
                'mean': observations.mean(axis=0),
                'std': observations.std(axis=0),
                'min': observations.min(axis=0),
                'max': observations.max(axis=0)
            }
            
            self.act_stats = {
                'mean': actions.mean(axis=0),
                'std': actions.std(axis=0),
                'min': actions.min(axis=0),
                'max': actions.max(axis=0)
            }
            
            print(f"Data statistics loaded from: {expert_data_path}")
            
        except Exception as e:
            print(f"Failed to load data statistics: {e}")
            # Use default normalization if expert data not available
            self.obs_stats = None
            self.act_stats = None
    
    def normalize_state(self, state):
        """Normalize state to [-1, 1] range (matching training normalization) with numerical stability"""
        if self.obs_stats is None:
            return state
        
        # Handle zero range to avoid division by zero
        obs_range = self.obs_stats['max'] - self.obs_stats['min']
        
        # For dimensions with zero variance (like pd_0), use the value directly without normalization
        state_norm = np.zeros_like(state)
        for i in range(len(state)):
            if obs_range[i] > 1e-8:  # Normal case: has variance
                state_norm[i] = (state[i] - self.obs_stats['min'][i]) / obs_range[i] * 2 - 1
            else:  # Zero variance case: use mean as normalized value
                # For constant dimensions, normalize to 0
                state_norm[i] = 0.0
                
        # Clamp to valid range to handle out-of-distribution inputs
        state_norm = np.clip(state_norm, -3.0, 3.0)  # Allow slight extrapolation
        
        return state_norm
    
    def denormalize_action(self, action_norm):
        """
        Enhanced action denormalization: matches the inverse transform of normalize_actions
        - Stiffness K1-K6: logarithmic inverse transform
        - Damping coefficient: square root inverse transform
        """
        if self.act_stats is None:
            # Use default ranges if statistics not available
            action_min = np.array([200, 200, 200, 50, 50, 50, 0.1])
            action_max = np.array([1000, 1000, 1000, 400, 400, 400, 5.0])
        else:
            action_min = self.act_stats['min']
            action_max = self.act_stats['max']
        
        action_denorm = action_norm.copy()
        
        for i in range(len(action_norm)):
            if i < 6:  # Stiffness K1-K6 - inverse logarithmic transform (CORRECTED!)
                epsilon = 1e-3
                act_min_safe = max(action_min[i], epsilon)
                act_max_safe = max(action_max[i], epsilon)
                
                log_min = np.log(act_min_safe)
                log_max = np.log(act_max_safe)
                log_range = log_max - log_min
                
                if log_range > 0:
                    log_value = action_norm[i] * log_range + log_min
                    action_denorm[i] = np.exp(log_value)
                    # Ensure within valid range
                    action_denorm[i] = np.clip(action_denorm[i], action_min[i], action_max[i])
                else:
                    action_denorm[i] = action_min[i]
            else:  # Damping coefficient - square root inverse transform
                sqrt_min = np.sqrt(action_min[i])
                sqrt_max = np.sqrt(action_max[i])
                sqrt_range = sqrt_max - sqrt_min
                
                if sqrt_range > 0:
                    sqrt_value = action_norm[i] * sqrt_range + sqrt_min
                    action_denorm[i] = np.square(sqrt_value)
                    # Ensure within valid range
                    action_denorm[i] = np.clip(action_denorm[i], action_min[i], action_max[i])
                else:
                    action_denorm[i] = action_min[i]
        
        return action_denorm
    
    def get_impedance_parameters(self, tracking_error, desired_position):
        """Get impedance parameters from AIRL policy"""
        # Construct state [e_t(3), pd_t(3)]
        state = np.concatenate([tracking_error.flatten(), desired_position.flatten()])
        
        # Normalize state
        state_norm = self.normalize_state(state)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        
        # Ensure policy is in evaluation mode for inference
        self.policy.eval()
        
        # Get policy output (normalized action)
        with torch.no_grad():
            action_norm = self.policy.get_deterministic_action(state_tensor)
            action_norm = action_norm.cpu().numpy()[0]
        
        # Denormalize to get actual impedance parameters
        impedance_params = self.denormalize_action(action_norm)
        
        # Extract parameters: [K1, K2, K3, K4, K5, K6, damping_ratio]
        K_diag = impedance_params[:6]
        damping_ratio = impedance_params[6]
        
        # Construct stiffness matrix (diagonal)
        K_matrix = np.diag(K_diag)
        
        # Construct damping matrix: D = damping_ratio * sqrt(K)
        D_diag = damping_ratio * np.sqrt(K_diag)
        D_matrix = np.diag(D_diag)
        
        return K_matrix, D_matrix, damping_ratio

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
                    'transition_factor': sample.transition_factor
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
                    'transition_factor': sample.transition_factor
                }
            elif target_time >= times[-1]:
                sample = self.buffer[-1] 
                return sample.desired_q.copy(), {
                    'pd_modified': sample.pd_modified.copy(),
                    'Rd_modified': sample.Rd_modified.copy(),
                    'x_admittance': sample.x_admittance.copy(),
                    'distance_to_hole': sample.distance_to_hole,
                    'stiffness_norm': sample.stiffness_norm,
                    'transition_factor': sample.transition_factor
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
                
                # SLERP for rotation matrices
                R_interp = self._slerp_rotation(sample1.Rd_modified, sample2.Rd_modified, alpha)
                
                return interpolated_q, {
                    'pd_modified': pd_interp,
                    'Rd_modified': R_interp,
                    'x_admittance': x_interp,
                    'distance_to_hole': dist_interp,
                    'stiffness_norm': stiff_interp,
                    'transition_factor': trans_interp
                }
                
    def _slerp_rotation(self, R1, R2, t):
        """Spherical linear interpolation for rotation matrices"""
        r1 = ScipyRotation.from_matrix(R1)
        r2 = ScipyRotation.from_matrix(R2)
        key_rots = ScipyRotation.concatenate([r1, r2])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        return slerp(t).as_matrix()

    def get_buffer_length(self) -> int:
        """Return the current number of samples in the interpolator buffer (thread-safe)."""
        with self.lock:
            return len(self.buffer)

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
    def __init__(self, task='pih', mode=None):
        self.n = 6
        self.xml_file = 'models/jaka_zu12/jaka_pih.xml'
        self.task = task
        
        self.duration = 2.0
            
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

        # Initialize AIRL Policy Manager for dynamic impedance parameters
        try:
            self.airl_policy_manager = AIRLPolicyManager(
                policy_path="script/models/airl/policy.pt",
                expert_data_path="data/expert_demonstration.pkl"
            )
            print("AIRL Policy Manager initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize AIRL Policy Manager: {e}")
            print("Falling back to manual impedance parameters")
            self.airl_policy_manager = None

        # Admittance control parameters (used in policy thread)
        self.d = 0.5
        self.Mc = np.diag([20.0, 20.0, 20.0, 5, 5, 5])
        
        self.Kc_far = np.diag([1000.0, 1000.0, 1000.0, 400.0, 400.0, 400.0])
        self.Kc_near = np.diag([200.0, 200.0, 200.0, 50.0, 50.0, 50.0])
        self.Kc = self.Kc_far.copy() 
        self.distance_threshold = 0.05  #
        
        self.current_Kc = self.Kc_far.copy()
        self.current_Dc = self.current_Kc * self.d
        self.current_damping_ratio = self.d  # Initialize with default damping ratio
        
        self.hole_position = np.array([0.0, -0.7, 0.12])  # Center of hole
        
        self.Dc = self.Kc * self.d

        # Admittance states (used in policy thread only)
        self.x_admittance = np.zeros(6)
        self.dx_admittance = np.zeros(6)
        self.ddx_admittance = np.zeros(6)
        self.admittance_lock = threading.Lock()

        # Trajectory recording
        self.trajectory_data = {
            'time': [], 'desired_pos': [], 'actual_pos': [], 'modified_pos': [],
            'desired_rot': [], 'actual_rot': [], 'modified_rot': [],
            'force': [], 'admittance_displacement': [], 'joint_angles': [],
            'distance_to_hole': [], 'stiffness_norm': [], 'transition_factor': [], 'damping_ratio': [],
            'K1': [], 'K2': [], 'K3': [], 'K4': [], 'K5': [], 'K6': []
        }
        
        # Store latest robot state for data recording
        self.latest_robot_state = None

        # Initialize trajectory functions
        self.pd_t, self.Rd_t, self.dpd_t, self.dRd_t, self.ddpd_t, self.ddRd_t = calculate_desired_pose_trajectory(self.task, self.duration)
        # Use shared helper in misc_func for initial configuration
        from misc_func import get_initial_joint_config
        self.initial_q = get_initial_joint_config(self.task, self.xml_file, IKArm, mode)
        
        # Shared references for thread access
        self.model = None
        self.data = None
        self.robot_state = None
        self.ik_solver = None
        self.previous_q = None

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
                        
                        error_6d = np.concatenate([pos_error, np.zeros(3)])
                        
                        external_wrench = robot_state.force_torque
                        if len(external_wrench) != 6:
                            if len(external_wrench) > 6:
                                external_wrench = external_wrench[:6]
                            else:
                                external_wrench = np.pad(external_wrench, (0, 6-len(external_wrench)))
                        
                        wrench_6d = external_wrench
                        
                        distance_to_hole, transition_factor = self.update_adaptive_stiffness(robot_state.current_position)
                        
                        self.ddx_admittance = np.linalg.solve(self.Mc, 
                            wrench_6d - self.current_Dc @ self.dx_admittance - self.current_Kc @ self.x_admittance)
                        
                        self.dx_admittance += self.ddx_admittance * 0.04
                        self.x_admittance += self.dx_admittance * 0.04
                        
                        pd_modified = pd_current + self.x_admittance[:3]
                        Rd_modified = Rd_current
                        
                        # Create target transform
                        Tep = np.eye(4)
                        Tep[:3, :3] = Rd_modified
                        Tep[:3, 3] = pd_modified
                    
                    # Solve IK using snapshot data
                    q_sol, success, iterations, error, jl_valid, solve_time = self.ik_solver.solve_ik_from_snapshot(
                        self.model, Tep, robot_state.q, robot_state.jacobian
                    )
                    
                    if success:
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
                            success=True
                        )
                        
                        self.interpolator.add_sample(policy_output)
                        
                        if robot_state.timestamp % 2.0 < 0.04:
                            kc_norm = np.linalg.norm(np.diag(self.current_Kc[:3, :3]))
                            policy_source = "AIRL" if self.airl_policy_manager is not None else "Manual"
                            self.get_logger().info(
                                f"Policy: t={robot_state.timestamp:.2f}s, solve_time={solve_time*1000:.1f}ms, "
                                f"dist_to_hole={distance_to_hole*100:.1f}cm, Kc_norm={kc_norm:.0f}, "
                                f"transition={transition_factor:.2f}, source={policy_source}"
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
                        alpha = 0.8
                        self.desired_q = alpha * interpolated_q + (1 - alpha) * self.desired_q
                        
                        # Record data
                        if current_time % 0.08 < 0.008 and interpolated_data is not None:
                            try:
                                if hasattr(self, 'latest_robot_state') and self.latest_robot_state is not None:
                                    pd_desired = self.pd_t(current_time).reshape(-1)
                                    Rd_desired = self.Rd_t(current_time)
                                    
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

                                    # Decide whether to use interpolated distance or compute from latest robot state.
                                    # Use interpolated_data only when the interpolator buffer has at least 2 samples
                                    # (so interpolation/extrapolation is meaningful). Otherwise compute distance
                                    # directly from the latest robot state to avoid recording the initial placeholder.
                                    try:
                                        buf_len = self.interpolator.get_buffer_length()
                                    except Exception:
                                        buf_len = 0

                                    if buf_len >= 2 and interpolated_data is not None and 'distance_to_hole' in interpolated_data:
                                        # Trust interpolated value when there are enough samples
                                        dist_to_hole = float(interpolated_data['distance_to_hole'])
                                    else:
                                        # Fallback: compute from main-thread robot snapshot (latest position)
                                        try:
                                            dist_to_hole = float(np.linalg.norm(self.latest_robot_state.current_position - self.hole_position))
                                        except Exception:
                                            # Last resort: use whatever interpolated value is available or zero
                                            dist_to_hole = interpolated_data.get('distance_to_hole') if interpolated_data is not None else 0.0

                                    self.trajectory_data['distance_to_hole'].append(dist_to_hole)
                                    self.trajectory_data['stiffness_norm'].append(interpolated_data['stiffness_norm'])
                                    self.trajectory_data['transition_factor'].append(interpolated_data['transition_factor'])
                                    self.trajectory_data['damping_ratio'].append(self.current_damping_ratio)

                                    # Record per-axis stiffness K1..K6. Prefer current_Kc if available (policy output),
                                    # otherwise fall back to controller's Kc diagonal.
                                    try:
                                        K_diag = np.diag(self.current_Kc)
                                    except Exception:
                                        try:
                                            K_diag = np.diag(self.Kc)
                                        except Exception:
                                            K_diag = np.zeros(6)

                                    for k_idx in range(6):
                                        self.trajectory_data[f'K{k_idx+1}'].append(float(K_diag[k_idx]))
                                    
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
        """Update Kc and Dc using AIRL policy or fallback to distance-based method"""
        distance_to_hole = np.linalg.norm(peg_position - self.hole_position)
        
        if self.airl_policy_manager is not None:
            try:
                current_time = self.current_time if hasattr(self, 'current_time') else 0.0
                pd_current = np.array(self.pd_t(current_time)).flatten()
                tracking_error = peg_position - pd_current
                
                Kc_policy, Dc_policy, damping_ratio = self.airl_policy_manager.get_impedance_parameters(
                    tracking_error, pd_current
                )
                
                self.current_Kc = Kc_policy
                self.current_Dc = Dc_policy
                self.current_damping_ratio = damping_ratio
                self.Kc = Kc_policy
                self.Dc = Dc_policy
                
                transition_sharpness = 50.0
                transition_factor = 1.0 / (1.0 + np.exp(-transition_sharpness * (distance_to_hole - self.distance_threshold)))
                
                return distance_to_hole, transition_factor
                
            except Exception as e:
                self.get_logger().error(f"AIRL policy error: {e}, falling back to manual control")
        
        # Fallback: manual distance-based adaptive stiffness
        transition_sharpness = 50.0
        transition_factor = 1.0 / (1.0 + np.exp(-transition_sharpness * (distance_to_hole - self.distance_threshold)))
        
        self.Kc = transition_factor * self.Kc_far + (1.0 - transition_factor) * self.Kc_near
        self.Dc = self.Kc * self.d
        
        self.current_Kc = self.Kc
        self.current_Dc = self.Dc
        self.current_damping_ratio = self.d  # Use default damping ratio for fallback
        
        return distance_to_hole, transition_factor

    def update_admittance_dynamics(self, Fe, dt):
        """Update admittance dynamics using current impedance parameters"""
        if Fe.ndim > 1:
            Fe = Fe.flatten()
        
        Mc_inv = np.linalg.inv(self.Mc)
        self.ddx_admittance = Mc_inv @ (Fe - self.current_Dc @ self.dx_admittance - self.current_Kc @ self.x_admittance)
        
        self.dx_admittance += self.ddx_admittance * dt
        self.x_admittance += self.dx_admittance * dt
        
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
        initial_distance = np.linalg.norm(initial_pd - self.hole_position)
        initial_policy_output = PolicyOutput(
            timestamp=0.0,
            desired_q=self.initial_q.copy(),
            pd_modified=initial_pd.copy(),
            Rd_modified=initial_Rd.copy(),
            x_admittance=np.zeros(6),
            distance_to_hole=initial_distance,
            stiffness_norm=np.linalg.norm(np.diag(self.Kc_far[:3, :3])),
            transition_factor=1.0,  # Start far, so use high stiffness
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
        
        # Log AIRL integration status
        if self.airl_policy_manager is not None:
            self.get_logger().info("AIRL Policy Integration: ACTIVE - Using learned impedance parameters")
        else:
            self.get_logger().info("AIRL Policy Integration: INACTIVE - Using manual impedance parameters")
        
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
                            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "jaka_end_effector")
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
                
                # Maintain simulation frequency
                elapsed = time.time() - step_start
                sleep_time = max(0, self.model.opt.timestep - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
        # Clean shutdown
        self.get_logger().info("Simulation finished")
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
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'log')
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate unique CSV filename with trajectory index
        csv_file = os.path.join(log_dir, f'trajectory_{self.task}_policy.csv')
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
                row.append(self.trajectory_data['damping_ratio'][i])
                # Append per-axis stiffness values K1..K6
                for k_idx in range(6):
                    row.append(self.trajectory_data.get(f'K{k_idx+1}', [0]*len(self.trajectory_data['time']))[i])
                row.extend(self.trajectory_data['joint_angles'][i])
                writer.writerow(row)
        
        self.get_logger().info(f"CSV trajectory data saved to: {csv_file}")

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


def test_airl_policy():
    """Test AIRL policy integration"""
    print("="*60)
    print("Testing AIRL Policy Integration")
    print("="*60)
    
    try:
        # Initialize AIRL policy manager
        airl_manager = AIRLPolicyManager(
            policy_path="script/models/airl/policy.pt",
            expert_data_path="data/expert_demonstration.pkl"
        )
        
        # Test with sample inputs
        test_cases = [
            (np.array([0.01, 0.02, -0.01]), np.array([0.0, -0.7, 0.35])),
            (np.array([0.05, -0.03, 0.02]), np.array([0.0, -0.7, 0.23])),
            (np.array([0.001, 0.001, -0.001]), np.array([0.0, -0.7, 0.12])),
        ]
        
        for i, (error, desired_pos) in enumerate(test_cases):
            print(f"\nTest case {i+1}:")
            print(f"  Tracking error: {error}")
            print(f"  Desired position: {desired_pos}")
            
            K_matrix, D_matrix, damping_ratio = airl_manager.get_impedance_parameters(error, desired_pos)
            
            print(f"  Stiffness (diagonal): {np.diag(K_matrix)[:3]} N/m, {np.diag(K_matrix)[3:]} Nm/rad")
            print(f"  Damping (diagonal): {np.diag(D_matrix)[:3]} N*s/m, {np.diag(D_matrix)[3:]} Nm*s/rad")
            print(f"  Damping ratio: {damping_ratio:.2f}")
            print(f"  Stiffness norm: {np.linalg.norm(np.diag(K_matrix)[:3]):.1f}")
            
        print(f"\nAIRL Policy Integration Test Passed!")
        return True
        
    except Exception as e:
        print(f"\nAIRL Policy Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(args=None):

    parser = argparse.ArgumentParser(description="AIRL Variable Admittance Control Simulator")
    
    parser.add_argument("--mode", type=str, default=None, help="Initial config mode: 'test' for fixed seed, otherwise random")

    parsed = parser.parse_args(args if args is not None else [])

    mode1 = parsed.mode

    print("="*60)
    print("AIRL Variable Admittance Control")
    print("="*60)
    
    # Test AIRL policy first
    if not test_airl_policy():
        print("\nAIRL integration test failed, but simulation will continue with fallback parameters")
    simulator = MujocoSimulator(task="pih", mode=mode1)
    
    print(f"\nAsynchronous Control Architecture:")
    print(f"- Policy (IK): {simulator.policy_frequency} Hz")
    print(f"- Controller: {simulator.controller_frequency} Hz") 
    print("- Simulation: Continuous, non-blocking")
    print("- Impedance Control: AIRL-Enhanced Variable Admittance")
    print("\nControls: Space = Pause/Resume")

    try:
        simulator.MujocoSim()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    import sys
    main(sys.argv[1:] if len(sys.argv) > 1 else None)