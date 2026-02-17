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
        
        self.max_force = max_force
        
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "jaka_end_effector")
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "attachment")
        
        try:
            self.hole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hole")
            # Need to forward kinematics first to get world position
            mujoco.mj_forward(self.model, self.data)
            self.hole_pos = self.data.xpos[self.hole_body_id].copy()
        except:
            self.hole_pos = np.array([0.0, -0.7, 0.12])

        self.arm_joint_indices = np.arange(6)
        
        obs_dim = 6 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
    
        self.k_range = [50.0, 1000.0]
        self.damping_ratio_range = [0.1, 5.0]

        self.trajectory_time = 0.0 
        self.trajectory_duration = 3.5  
        self.start_position = None  
        self.target_position = None 
        
        self.initial_qpos = np.array([-1.75916382, 1.27408681, -2.06908278,
                                      2.35226387, 1.57079097, 1.38243476])
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        # Data logging
        self.trajectory_log = []
        
    def _get_obs(self):
        """Get current observation: [e_t, pd_t] for Variable Admittance Control."""
        # Get current end-effector position (actual position p)
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        
        # Calculate desired position pd_t based on current trajectory time
        pd_t = self._get_desired_position(self.trajectory_time)
        
        # Tracking error e_t = actual_position - desired_position (p - pd)
        tracking_error = ee_pos - pd_t
        
        # Combined observation: [e_t(3), pd_t(3)] = 6D
        obs = np.concatenate([tracking_error, pd_t]).astype(np.float32)
        
        return obs
    
    def _get_desired_position(self, t):
        """
        Calculate desired position pd_t at time t using a smooth trajectory
        from start_position to target_position (hole)
        """
        if self.start_position is None or self.target_position is None:
            return np.array([0.0, -0.7, 0.12])  # Default position
        
        # Normalize time to [0, 1]
        t_norm = min(t / self.trajectory_duration, 1.0)
        
        # Use smooth S-curve trajectory (quintic polynomial)
        # s(t) = 6t^5 - 15t^4 + 10t^3
        s = 6 * t_norm**5 - 15 * t_norm**4 + 10 * t_norm**3
        
        # Interpolate between start and target position
        pd_t = self.start_position + s * (self.target_position - self.start_position)
        
        return pd_t
    
    def _rotation_matrix_to_axis_angle(self, rotation_matrix):
        """Convert rotation matrix to axis-angle representation."""
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rotation_matrix)
        axis_angle = r.as_rotvec()
        return axis_angle
    
    def _scale_action(self, action):
        """Scale normalized action [0, 1] to actual impedance parameters."""

        k_params = action[0:6] * (self.k_range[1] - self.k_range[0]) + self.k_range[0]
        damping_ratio = action[6] * (self.damping_ratio_range[1] - self.damping_ratio_range[0]) + self.damping_ratio_range[0]
        
        return k_params, damping_ratio
    
    def _check_success(self, obs):
        """Check if peg is successfully inserted into hole."""
        tracking_error = obs[0:3] 
        desired_position = obs[3:6] 
        
        target_distance = np.linalg.norm(desired_position - self.target_position)
        tracking_error_norm = np.linalg.norm(tracking_error)
        
        return target_distance < 0.02 and tracking_error_norm < 0.01
    
    def _check_failure(self, obs):
        """Check if episode should terminate due to failure."""
        tracking_error = obs[0:3]
        
        tracking_error_norm = np.linalg.norm(tracking_error)
        if tracking_error_norm > 0.5:  
            return True
            
        joint_pos = self.data.qpos[self.arm_joint_indices]
        joint_limits_low = self.model.jnt_range[self.arm_joint_indices, 0]
        joint_limits_high = self.model.jnt_range[self.arm_joint_indices, 1]
        if np.any(joint_pos < joint_limits_low) or np.any(joint_pos > joint_limits_high):
            return True
        
        return False
    
    def step(self, action):
        """Execute one step in the environment with Variable Admittance Control."""
        self.trajectory_time += self.control_dt
        
        k_params, damping_ratio = self._scale_action(action)

        obs_current = self._get_obs()
        tracking_error = obs_current[0:3]
        desired_position = obs_current[3:6]

        K = np.diag(k_params[:3])
        M = np.diag([200.0, 200.0, 200.0])
        D = damping_ratio * np.sqrt(K)

        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)
        qvel = self.data.qvel.copy()
        ee_vel = jacp @ qvel

        pd_t = desired_position
        pd_dot = (pd_t - ee_pos) / self.control_dt

        e_pos = ee_pos - pd_t
        e_vel = ee_vel - pd_dot

        F_external = -K @ e_pos - D @ e_vel

        A = M / (self.control_dt * self.control_dt) + D / (2.0 * self.control_dt)
        b = F_external - K @ e_pos
        e_admittance = np.linalg.solve(A, b)

        pd_modified = pd_t + e_admittance

        nv = self.model.nv
        jacp = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)

        J_pos = jacp[:, :6]

        position_error = pd_modified - ee_pos
        desired_force = K @ position_error
        tau = J_pos.T @ desired_force
        self.data.ctrl[:6] = tau

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        
        reward = 0.0
        
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
            'trajectory_time': self.trajectory_time,
            'impedance_params': {
                'k_params': k_params,
                'damping_ratio': damping_ratio
            },
            'admittance_modification': e_admittance,
            'desired_position': desired_position,
            'modified_desired_position': pd_modified,
            'tracking_error': tracking_error
        }

        self.trajectory_log.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'tracking_error': tracking_error.copy(),
            'desired_position': desired_position.copy(),
            'admittance_modification': e_admittance.copy(),
            'trajectory_time': self.trajectory_time,
            'is_success': success,
            'is_failure': failure
        })
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.initial_qpos = np.array([-1.759, 1.245, -2.084, 2.397, 1.571, 1.382])
        self.data.qpos[self.arm_joint_indices] = self.initial_qpos

        if seed is not None:
            np.random.seed(seed)
        self.data.qpos[self.arm_joint_indices] += np.random.uniform(-0.02, 0.02, size=6)

        mujoco.mj_forward(self.model, self.data)
        try:
            self.hole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            self.hole_pos = self.data.xpos[self.hole_body_id].copy()
            self.hole_pos[2] = 0.02
        except:
            self.hole_pos = np.array([0.0, -0.7, 0.12]) 
        
        self.target_position = self.hole_pos.copy()

        self.start_position = self.data.site_xpos[self.ee_site_id].copy()

        self.trajectory_time = 0.0

        self.current_step = 0
        self.trajectory_log = []

        obs = self._get_obs()
        info = {
            'hole_position': self.hole_pos.copy(),
            'start_position': self.start_position.copy(),
            'target_position': self.target_position.copy(),
            'trajectory_duration': self.trajectory_duration
        }
        
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
