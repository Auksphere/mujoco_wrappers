import numpy as np

import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

class UR10Env(gym.Env):
    def __init__(
        self,
        xml_path,
        seed:int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._viewer = MujocoRenderer(self._model, self._data)

        # Define observation spaces
        self.observation_space = spaces.Dict(
            {
                "arm": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                    "joint_pos": spaces.Box(low = np.array([self._model.jnt_range[i][0] for i in range(1, 7)]),
                                            high = np.array([self._model.jnt_range[i][1] for i in range(1, 7)]),
                                            dtype=np.float32),
                }),
                "hand": spaces.Box(low = np.array([self._model.jnt_range[i][0] for i in hand_joint_indices]),
                                   high = np.array([self._model.jnt_range[i][1] for i in hand_joint_indices]),
                                   dtype=np.float32),
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),
                    # "shape": spaces.Box(-5, 5, shape=(2,), dtype=np.float32),
                }),
            }
        )

        # Define action spaces
        self.action_space = spaces.Dict(
            {
                "arm": spaces.Box(arm_low, arm_high, shape=(4,), dtype=np.float32)
            }
        )
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq,), dtype=np.float32)

    def step(self, action):
        mujoco.mj_step(self.model, self.data, action)
        obs = self.data.qpos.copy()
        reward = 0.0  # Define your reward function here
        done = False  # Define your termination condition here
        info = {}
        return obs, reward, done, info

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self.data.qpos.copy()

    def render(self, mode='human'):
        if mode == 'human':
            mujoco.viewer.launch_passive_viewer(self.viewer)
        else:
            raise NotImplementedError("Only 'human' mode is supported for rendering.")
