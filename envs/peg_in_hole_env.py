#!/usr/bin/env python3
"""
Peg-in-Hole Environment for Variable Impedance Control Learning

This environment is designed for learning variable impedance control policies
using Adversarial Inverse Reinforcement Learning (AIRL) as described in:
"Learning Variable Impedance Control via Inverse Reinforcement Learning for Force-Related Tasks"
by Zhang et al., IEEE RA-L 2021.

The environment provides:
- State: joint positions, velocities, end-effector pose, force/torque readings
- Action: impedance parameters (Kp, Kd) for each DOF
- Reward: shaped by distance to goal and force constraints
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
import os


class PegInHoleEnv(gym.Env):
    """
    Peg-in-Hole environment with variable impedance control.
    
    The task is to insert a cylindrical peg into a hole with tight clearance.
    Success requires appropriate contact force management through impedance control.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        xml_path: str = "models/jaka_zu12/jaka_pih.xml",
        control_dt: float = 0.008,  # 25Hz control frequency
        physics_dt: float = 0.001,  # 1kHz simulation frequency
        max_episode_steps: int = 500,
        render_mode: str = None,
        success_threshold: float = 0.01,  # 1cm for successful insertion
        max_force: float = 50.0,  # Maximum allowed force in Newtons
    ):
        super().__init__()
        
        # Load MuJoCo model
        xml_full_path = os.path.join(os.path.dirname(__file__), '..', xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_full_path)
        self.data = mujoco.MjData(self.model)
        
        # Timing parameters
        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self.n_substeps = int(control_dt / physics_dt)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Task parameters
        self.success_threshold = success_threshold
        self.max_force = max_force
        
        # Get important model IDs
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "jaka_end_effector")
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "attachment")
        
        # Get hole position from body (not geom, as hole is a body with multiple geoms)
        try:
            self.hole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hole")
            # Need to forward kinematics first to get world position
            mujoco.mj_forward(self.model, self.data)
            self.hole_pos = self.data.xpos[self.hole_body_id].copy()
        except:
            # Default hole position if not found
            self.hole_pos = np.array([0.0, -0.7, 0.1])
        
        # Joint indices (first 6 joints are the robot arm)
        self.arm_joint_indices = np.arange(6)
        
        # Define observation space for IRL
        # State: [pose_error(3), orientation_error(3), pose_velocity_error(3), orientation_velocity_error(3)]
        # Total: 6D pose error + 6D velocity error = 12D
        obs_dim = 6 + 6  # pose error + velocity error
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Define action space for IRL
        # Action: [K1, K2, K3, K4, K5, K6, damping_ratio_d] = 7D impedance parameters
        # K1-K3: linear stiffness, K4-K6: angular stiffness, d: damping ratio
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # Impedance parameter ranges for IRL action scaling (reduced for stability)
        # K1-K3: Linear stiffness (N/m)
        self.k_linear_range = [50.0, 800.0]  # Reduced from [100.0, 2000.0]
        # K4-K6: Angular stiffness (Nm/rad) 
        self.k_angular_range = [5.0, 200.0]  # Reduced from [10.0, 500.0]
        # Damping ratio d (dimensionless, typically 0.1 to 2.0)
        self.damping_ratio_range = [0.3, 1.5]  # Narrowed from [0.1, 2.0]
        
        # Target pose for hole insertion (updated based on your XML modifications)
        # This will be set during reset based on actual hole position
        self.target_pose = np.eye(4)
        self.target_velocity = np.zeros(6)  # Target velocity is zero (static target)
        
        # Initial configuration (from jaka_pih.xml or pre-computed)
        self.initial_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        # Data logging
        self.trajectory_log = []
        
    def _get_obs(self):
        """Get current observation: Cartesian pose and velocity errors for IRL."""
        # Get current end-effector pose
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        
        # Get current end-effector velocity
        ee_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, 
            self.ee_site_id, ee_vel, 0
        )
        
        # Target position: hole position with small insertion depth
        target_pos = self.hole_pos.copy()
        target_pos[2] -= 0.02  # Insert 2cm into hole
        
        # Target orientation: vertical pointing down
        target_rot = np.array([[0, 1, 0],
                              [1, 0, 0], 
                              [0, 0, -1]])
        
        # Compute pose errors (target - current for proper control direction)
        pos_error = target_pos - ee_pos
        
        # Rotation error using axis-angle representation
        rot_error_mat = target_rot.T @ ee_mat
        rot_error = self._rotation_matrix_to_axis_angle(rot_error_mat)
        
        # Combine pose error
        pose_error = np.concatenate([pos_error, rot_error])
        
        # Velocity error (target velocity is zero)
        velocity_error = ee_vel - self.target_velocity
        
        # Combined observation: [pose_error(6), velocity_error(6)]
        obs = np.concatenate([pose_error, velocity_error]).astype(np.float32)
        
        return obs
    
    def _rotation_matrix_to_axis_angle(self, rotation_matrix):
        """Convert rotation matrix to axis-angle representation."""
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rotation_matrix)
        axis_angle = r.as_rotvec()
        return axis_angle
    
    def _scale_action(self, action):
        """Scale normalized action [0, 1] to actual impedance parameters for IRL."""
        # K1-K3: Linear stiffness
        k_linear = action[0:3] * (self.k_linear_range[1] - self.k_linear_range[0]) + self.k_linear_range[0]
        # K4-K6: Angular stiffness  
        k_angular = action[3:6] * (self.k_angular_range[1] - self.k_angular_range[0]) + self.k_angular_range[0]
        # Damping ratio d
        damping_ratio = action[6] * (self.damping_ratio_range[1] - self.damping_ratio_range[0]) + self.damping_ratio_range[0]
        
        return k_linear, k_angular, damping_ratio
    
    def _compute_reward(self, obs, action):
        """
        Compute reward based on pose and velocity errors for IRL training.
        Observation format: [pose_error(6), velocity_error(6)]
        """
        # Extract pose and velocity errors from observation
        pose_error = obs[0:6]     # [pos_error(3), rot_error(3)]
        velocity_error = obs[6:12] # [vel_error(6)]
        
        pos_error = pose_error[0:3]
        rot_error = pose_error[3:6]
        
        # Calculate distance metrics
        pos_distance = np.linalg.norm(pos_error)
        rot_distance = np.linalg.norm(rot_error)
        vel_magnitude = np.linalg.norm(velocity_error)
        
        # Z-direction error (critical for insertion)
        z_error = abs(pos_error[2])
        xy_error = np.linalg.norm(pos_error[:2])
        
        # === Staged rewards based on insertion phase ===
        scale = 0.1
        scale2 = 1.0
        
        # Basic distance reward
        total_distance = pos_distance + 0.5 * rot_distance
        reward = -scale * np.clip(total_distance, 0, 1)
        
        # Stage 1: Close approach reward (within 5cm and Z < 4cm)
        if total_distance < 0.05 and z_error < 0.04:
            reward = scale2 * (0.04 - z_error)
        
        # Stage 2: Successful insertion reward (within 2cm and Z < 1cm)  
        if total_distance < 0.02 and z_error < 0.01:
            reward = 3.0  # High reward for successful insertion
        
        # Velocity penalty for excessive motion
        if vel_magnitude > 0.1:
            reward -= 0.01 * vel_magnitude
            
        return reward
    
    def _check_success(self, obs):
        """Check if peg is successfully inserted into hole using pose errors."""
        pose_error = obs[0:6]
        pos_error = pose_error[0:3]
        rot_error = pose_error[3:6]
        
        # Success criteria: position error < 2cm and rotation error < 0.2 rad
        pos_distance = np.linalg.norm(pos_error)
        rot_distance = np.linalg.norm(rot_error)
        z_error = abs(pos_error[2])
        
        # More reasonable success criteria
        return pos_distance < 0.02 and rot_distance < 0.2 and z_error < 0.015
    
    def _check_failure(self, obs):
        """Check if episode should terminate due to failure."""
        pose_error = obs[0:6]
        pos_error = pose_error[0:3]
        
        # More permissive failure condition - only fail if extremely far (> 50cm)
        pos_distance = np.linalg.norm(pos_error)
        if pos_distance > 0.5:  # Increased from 0.2 to 0.5
            return True
            
        # Check joint limits by getting current joint positions
        joint_pos = self.data.qpos[self.arm_joint_indices]
        joint_limits_low = self.model.jnt_range[self.arm_joint_indices, 0]
        joint_limits_high = self.model.jnt_range[self.arm_joint_indices, 1]
        if np.any(joint_pos < joint_limits_low) or np.any(joint_pos > joint_limits_high):
            return True
        
        return False
    
    def step(self, action):
        """Execute one step in the environment with IRL impedance control."""
        # Scale action to impedance parameters
        k_linear, k_angular, damping_ratio = self._scale_action(action)
        
        # Construct impedance matrices (K and D)
        K = np.diag(np.concatenate([k_linear, k_angular]))
        D = damping_ratio * 2 * np.sqrt(K)  # Critical damping relationship
        
        # Get current observation for control
        obs_current = self._get_obs()
        pose_error = obs_current[0:6]
        velocity_error = obs_current[6:12]
        
        # Impedance control: F = K*e + D*e_dot (error is target - current)
        desired_wrench = K @ pose_error + D @ velocity_error
        
        # Convert to joint torques using Jacobian transpose
        nv = self.model.nv
        jacp = np.zeros((3, nv))  # Position jacobian
        jacr = np.zeros((3, nv))  # Rotation jacobian
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        
        # Combine jacobians for first 6 joints (arm only)
        J = np.vstack([jacp[:3, :6], jacr[:3, :6]])
        
        # Apply joint torques
        tau = J.T @ desired_wrench
        self.data.ctrl[:6] = tau
        
        # Run simulation for n_substeps
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Get new observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward(obs, action)
        
        # Check termination
        success = self._check_success(obs)
        failure = self._check_failure(obs)
        self.current_step += 1
        
        terminated = success or failure
        truncated = self.current_step >= self.max_episode_steps
        
        # Info dictionary
        info = {
            'is_success': success,
            'is_failure': failure,
            'episode_step': self.current_step,
            'impedance_params': {
                'k_linear': k_linear,
                'k_angular': k_angular, 
                'damping_ratio': damping_ratio
            }
        }
        
        # Log trajectory data for IRL
        self.trajectory_log.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': reward,
            'impedance_K': K.copy(),
            'impedance_D': D.copy(),
            'pose_error': pose_error.copy(),
            'velocity_error': velocity_error.copy()
        })
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint configuration - get from hole insertion starting position
        # This should be a reasonable starting pose above the hole
        self.initial_qpos = np.array([-1.759, 1.245, -2.084, 2.397, 1.571, 1.382])  # Based on your trajectory
        self.data.qpos[self.arm_joint_indices] = self.initial_qpos
        
        # Add small random perturbations for variability
        if seed is not None:
            np.random.seed(seed)
        self.data.qpos[self.arm_joint_indices] += np.random.uniform(-0.02, 0.02, size=6)
        
        # Update hole position based on current model state
        mujoco.mj_forward(self.model, self.data)
        try:
            self.hole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            self.hole_pos = self.data.xpos[self.hole_body_id].copy()
            self.hole_pos[2] = 0.02  # Set hole center based on your modifications
        except:
            self.hole_pos = np.array([0.0, -0.7, 0.02])  # Default based on your XML
            
        # Set target pose for hole insertion
        self.target_pose[:3, 3] = self.hole_pos.copy()
        self.target_pose[:3, 3][2] -= 0.02  # Target 2cm into hole
        self.target_pose[:3, :3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # Vertical down
        
        # Reset counters
        self.current_step = 0
        self.trajectory_log = []
        
        # Get initial observation
        obs = self._get_obs()
        info = {'hole_position': self.hole_pos.copy()}
        
        return obs, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # TODO: Implement offscreen rendering
            pass
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def get_trajectory_log(self):
        """Return logged trajectory data."""
        return self.trajectory_log


# Register environment
gym.register(
    id='PegInHole-v0',
    entry_point='envs.peg_in_hole_env:PegInHoleEnv',
    max_episode_steps=500,
)
