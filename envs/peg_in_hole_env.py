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
        control_dt: float = 0.04,  # 25Hz control frequency
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
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        
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
        
        # Define observation space
        # State: [joint_pos(6), joint_vel(6), ee_pos(3), ee_quat(4), 
        #         ee_vel(6), force(3), torque(3), hole_pos(3)]
        obs_dim = 6 + 6 + 3 + 4 + 6 + 3 + 3 + 3  # = 34
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Define action space
        # Action: [Kp_linear(3), Kd_linear(3), Kp_angular(3), Kd_angular(3)]
        # Normalized to [0, 1] range, will be scaled to actual impedance ranges
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Impedance parameter ranges (N/m for linear, Nm/rad for angular)
        self.kp_linear_range = [100.0, 2000.0]
        self.kd_linear_range = [10.0, 200.0]
        self.kp_angular_range = [50.0, 500.0]
        self.kd_angular_range = [5.0, 100.0]
        
        # Initial configuration (from jaka_pih.xml or pre-computed)
        self.initial_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        # Data logging
        self.trajectory_log = []
        
    def _get_obs(self):
        """Get current observation."""
        # Joint positions and velocities
        joint_pos = self.data.qpos[self.arm_joint_indices].copy()
        joint_vel = self.data.qvel[self.arm_joint_indices].copy()
        
        # End-effector pose
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        ee_quat = R.from_matrix(ee_mat).as_quat()  # [x, y, z, w]
        
        # End-effector velocity (linear and angular)
        ee_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, 
            self.ee_site_id, ee_vel, 0
        )
        
        # Force/torque sensor readings
        try:
            fx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor_fx")
            fy_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor_fy")
            fz_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor_fz")
            mx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor_mx")
            my_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor_my")
            mz_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor_mz")
            
            force = np.array([
                self.data.sensordata[self.model.sensor_adr[fx_id]],
                self.data.sensordata[self.model.sensor_adr[fy_id]],
                self.data.sensordata[self.model.sensor_adr[fz_id]]
            ])
            torque = np.array([
                self.data.sensordata[self.model.sensor_adr[mx_id]],
                self.data.sensordata[self.model.sensor_adr[my_id]],
                self.data.sensordata[self.model.sensor_adr[mz_id]]
            ])
        except:
            # Fallback if sensors not found
            force = np.zeros(3)
            torque = np.zeros(3)
        
        # Hole position (goal)
        hole_pos = self.hole_pos.copy()
        
        # Concatenate observation
        obs = np.concatenate([
            joint_pos, joint_vel, ee_pos, ee_quat, 
            ee_vel, force, torque, hole_pos
        ]).astype(np.float32)
        
        return obs
    
    def _scale_action(self, action):
        """Scale normalized action [0, 1] to actual impedance parameters."""
        kp_linear = action[0:3] * (self.kp_linear_range[1] - self.kp_linear_range[0]) + self.kp_linear_range[0]
        kd_linear = action[3:6] * (self.kd_linear_range[1] - self.kd_linear_range[0]) + self.kd_linear_range[0]
        kp_angular = action[6:9] * (self.kp_angular_range[1] - self.kp_angular_range[0]) + self.kp_angular_range[0]
        kd_angular = action[9:12] * (self.kd_angular_range[1] - self.kd_angular_range[0]) + self.kd_angular_range[0]
        
        return kp_linear, kd_linear, kp_angular, kd_angular
    
    def _compute_reward(self, obs, action):
        """
        Compute reward based on GIC_Learning_public successful strategy:
        - Distance to goal (both translation and rotation)
        - Staged rewards for different phases
        - Success bonus
        - Optional force penalty
        """
        # Extract relevant quantities from observation
        ee_pos = obs[12:15]     # indices 12-14 are ee_pos
        ee_quat = obs[15:19]    # indices 15-18 are ee_quat
        force = obs[25:28]      # indices 25-27 are force
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        ee_rot = R.from_quat(ee_quat).as_matrix()
        
        # Define desired orientation (vertical pointing down)
        Rd = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, -1]])
        
        # Calculate position and rotation errors (like in GIC_Learning_public)
        pos_error = ee_pos - self.hole_pos
        rot_error = np.trace(np.eye(3) - Rd.T @ ee_rot)
        trans_error = 0.5 * np.dot(pos_error, pos_error)
        
        # Combined distance metric
        total_distance = np.sqrt(rot_error + trans_error)
        
        # Z-direction error (critical for insertion)
        z_error = abs(pos_error[2])
        xy_error = np.sqrt(pos_error[0]**2 + pos_error[1]**2)
        
        # === GIC-STYLE STAGED REWARDS ===
        scale = 0.1
        scale2 = 1.0
        
        # Basic distance reward (always applied)
        reward = -scale * np.clip(total_distance, 0, 1)
        
        # Stage 1: Close approach reward (within 20cm and Z < 4cm)
        if total_distance < 0.2 and z_error < 0.04:
            reward = scale2 * (0.04 - z_error)
        
        # Stage 2: Successful insertion reward (within 10cm and Z < 2.6cm)
        if total_distance < 0.1 and z_error < 0.026:
            reward = 3.0  # High reward for successful insertion
        
        # Force penalty (optional, like in GIC with force_penalty version)
        force_magnitude = np.linalg.norm(force)
        if xy_error > 0.0002 and force_magnitude > 0:
            # Only penalize forces when not well-aligned
            reward -= 0.005 * abs(force[2])  # Penalize Z-force specifically
        
        return reward
    
    def _check_success(self, obs):
        """Check if peg is successfully inserted into hole using GIC criteria."""
        ee_pos = obs[12:15]
        ee_quat = obs[15:19]
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        ee_rot = R.from_quat(ee_quat).as_matrix()
        
        # Define desired orientation
        Rd = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, -1]])
        
        # Calculate errors
        pos_error = ee_pos - self.hole_pos
        rot_error = np.trace(np.eye(3) - Rd.T @ ee_rot)
        trans_error = 0.5 * np.dot(pos_error, pos_error)
        total_distance = np.sqrt(rot_error + trans_error)
        
        # Success criteria from GIC: distance < 0.1 and Z error < 2.6cm
        z_error = abs(pos_error[2])
        return total_distance < 0.1 and z_error < 0.026
    
    def _check_failure(self, obs):
        """Check if episode should terminate due to failure."""
        # Excessive force
        force = obs[25:28]
        if np.linalg.norm(force) > 2 * self.max_force:
            return True
        
        # Joint limits
        joint_pos = obs[0:6]
        joint_limits_low = self.model.jnt_range[self.arm_joint_indices, 0]
        joint_limits_high = self.model.jnt_range[self.arm_joint_indices, 1]
        if np.any(joint_pos < joint_limits_low) or np.any(joint_pos > joint_limits_high):
            return True
        
        return False
    
    def step(self, action):
        """Execute one step in the environment."""
        # Scale action to impedance parameters
        kp_linear, kd_linear, kp_angular, kd_angular = self._scale_action(action)
        
        # Construct impedance matrices
        Kp = np.diag(np.concatenate([kp_linear, kp_angular]))
        Kd = np.diag(np.concatenate([kd_linear, kd_angular]))
        
        # Implement impedance control
        # Get current end-effector state
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        ee_quat = self._mat_to_quat(ee_mat)
        
        # Define desired end-effector pose (move towards hole)
        desired_pos = self.hole_pos.copy()
        desired_quat = np.array([0, 1, 0, 0])  # Pointing down
        
        # Compute pose error
        pos_error = ee_pos - desired_pos
        quat_error = self._quat_error(ee_quat, desired_quat)
        
        # Get end-effector velocity
        ee_vel_linear = np.zeros(3)  # Simplified - could compute from jacobian
        ee_vel_angular = np.zeros(3)
        
        # Impedance control force/torque
        pose_error_6d = np.concatenate([pos_error, quat_error])
        vel_error_6d = np.concatenate([ee_vel_linear, ee_vel_angular])
        
        # Desired wrench in end-effector frame
        desired_wrench = -Kp @ pose_error_6d - Kd @ vel_error_6d
        
        # Convert to joint torques using Jacobian transpose
        nv = self.model.nv
        jacp = np.zeros((3, nv))  # Position jacobian
        jacr = np.zeros((3, nv))  # Rotation jacobian
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        
        # Combine jacobians
        J = np.vstack([jacp[:3, :6], jacr[:3, :6]])  # Use only first 6 columns for arm joints
        
        # Apply joint torques
        tau = J.T @ desired_wrench
        self.data.ctrl[:6] = tau
        
        # Run simulation for n_substeps
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
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
        }
        
        # Log trajectory
        self.trajectory_log.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': reward,
            'Kp': Kp.copy(),
            'Kd': Kd.copy(),
        })
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint configuration
        self.data.qpos[self.arm_joint_indices] = self.initial_qpos
        
        # Add small random perturbations
        if seed is not None:
            np.random.seed(seed)
        self.data.qpos[self.arm_joint_indices] += np.random.uniform(-0.05, 0.05, size=6)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset counters
        self.current_step = 0
        self.trajectory_log = []
        
        # Get initial observation
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def _mat_to_quat(self, rotation_matrix):
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # Returns (x, y, z, w)
        return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to (w, x, y, z)
    
    def _quat_error(self, q_current, q_desired):
        """Compute quaternion error as axis-angle representation."""
        # Quaternion error: q_error = q_desired * q_current^(-1)
        q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        q_error = self._quat_multiply(q_desired, q_current_conj)
        
        # Convert to axis-angle (use vector part scaled by 2)
        if q_error[0] < 0:
            q_error = -q_error
        return 2.0 * q_error[1:4]
    
    def _quat_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
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
