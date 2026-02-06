import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
import numpy as np
import sympy as sp
import sys
import os
from misc_func import calculate_desired_pose_trajectory

# Add project root to Python path to allow importing from 'controllers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
# Direct import to avoid loading all controllers dependencies
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


class MujocoIKControllerNode(Node):
    def __init__(self, task='sphere'):
        super().__init__('mujoco_ik_controller_node')
        self.n = 6
        self.xml_file = 'models/jaka_zu12/jaka_admittance.xml'
        self.task = task  # Task type: 'regulation', 'circle', 'line', 'sphere'
        
        # Initial joint configuration will be set based on task
        self.initial_q = None
        
        # Trajectory parameters
        self.duration = 10.0  # Total trajectory duration in seconds
        self.current_time = 0.0
        self.control_timestep = 0.008  # Update control every 8ms (125Hz)
        self.last_control_time = 0.0
        
        # Paused state
        self.paused = False

        # PD joint controller
        self.desired_q = np.array([0.0] * self.n)
        self.desired_velocity = np.array([0.0] * self.n)
        self.feedforward_torque = np.array([0.0] * self.n)

        self.k_p = np.array([400, 400, 400, 50, 25, 5])
        self.k_d = np.array([50, 50, 50, 3, 2, 1])

        self.max_joint_vel = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        self.max_joint_acc = np.array([5.0, 5.0, 5.0, 8.0, 8.0, 8.0])


        # Initialize trajectory functions and initial joint configuration
        self.pd_t, self.Rd_t, self.dpd_t, self.dRd_t, self.ddpd_t, self.ddRd_t = calculate_desired_pose_trajectory(self.task, self.duration)
        self.initial_q = self.get_initial_joint_config()
        
        # Initialize IK solver
        self.ik_solver = None
        
        # Previous joint angles for warm start
        self.previous_q = self.initial_q.copy()
        
        self.get_logger().info(f"MuJoCo IK Controller Node initialized with task: {self.task}")
    

    def get_initial_joint_config(self):
        """
        Returns the initial joint configuration based on task type.
        These values are pre-computed via IK to match trajectory starting points.
        All configurations have been computed with precision tol=1e-4, see in find_init.py.
        """

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
        
        if self.task in initial_configs:
            return initial_configs[self.task]
        else:
            # Default fallback configuration
            self.get_logger().warn(f"Unknown task '{self.task}', using default configuration")
            return np.array([0.0, np.pi/2, 0.0, np.pi/2, 0.0, 0.0])

    def MujocoSim(self):
        """
        Runs the MuJoCo simulation with real-time IK trajectory generation and control.
        """
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)
        
        # Initialize IK solver with higher precision
        # Since we have accurate initial position, we can use stricter tolerance
        self.ik_solver = IKArm(solver_type='QP', tol=1.5e-6, ilimit=5000)
        
        # Set initial joint positions
        data.qpos[:self.n] = self.initial_q
        self.previous_q = self.initial_q.copy()
        
        # Reset time
        self.current_time = 0.0
        self.last_control_time = 0.0
        
        self.get_logger().info(f"Starting trajectory execution for {self.duration} seconds...")
        self.get_logger().info(f"Control frequency: {1.0/self.control_timestep:.1f} Hz")
        
        with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
            while viewer.is_running() and self.current_time < self.duration:
                step_start = time.time()

                if not self.paused:
                    # Only update control commands at the specified control frequency
                    if self.current_time - self.last_control_time >= self.control_timestep:
                        # Calculate desired end-effector pose for current time
                        pd_current = np.array(self.pd_t(self.current_time)).flatten()
                        Rd_current = np.array(self.Rd_t(self.current_time)).reshape(3, 3)
                        
                        # Construct the target pose matrix (Tep)
                        Tep = np.eye(4)
                        Tep[:3, :3] = Rd_current
                        Tep[:3, 3] = pd_current
                        
                        # Solve IK for desired joint angles
                        q_sol, success, iterations, error, jl_valid, solve_time = self.ik_solver.solve(
                            model, data, Tep, self.previous_q
                        )
                        
                        if success:
                            # position_error = q_sol - data.qpos
                            # velocity_error = self.desired_velocity - data.qvel

                            # 计算期望速度(通过有限差分近似)
                            dt = self.control_timestep
                            desired_vel = (q_sol - self.desired_q) / dt
                            
                            # 限制速度变化率(加速度限制)
                            acc = (desired_vel - self.desired_velocity) / dt
                            acc = np.clip(acc, -self.max_joint_acc, self.max_joint_acc)
                            desired_vel = self.desired_velocity + acc * dt
                            
                            # 限制最大速度
                            desired_vel = np.clip(desired_vel, -self.max_joint_vel, self.max_joint_vel)
                            
                            # 平滑目标位置更新(低通滤波)
                            alpha = 0.5  # 滤波系数 (0-1, 越小越平滑)
                            self.desired_q = alpha * q_sol + (1 - alpha) * self.desired_q
                            
                            # 计算PD控制律
                            position_error = self.desired_q - data.qpos[:self.n]
                            velocity_error = desired_vel - data.qvel[:self.n]
                    
                            pd_torques = self.k_p * position_error + self.k_d * velocity_error

                            # Set control commands to desired joint angles
                            # data.ctrl[:self.n] = pd_torques + self.feedforward_torque
                            data.ctrl[:] = q_sol
                            self.previous_q = q_sol.copy()
                        else:
                            # If IK fails, use previous solution
                            self.get_logger().warn(
                                f"IK failed at t={self.current_time:.3f}s, error={error:.6f}, "
                                f"using previous solution"
                            )
                            data.ctrl[:self.n] = self.previous_q
                        
                        self.last_control_time = self.current_time
                    
                    # Step the simulation
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    
                    # Advance time by timestep
                    self.current_time += model.opt.timestep

                # Maintain simulation frequency
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                rclpy.spin_once(self, timeout_sec=0.0)
            
            self.get_logger().info("Trajectory execution finished.")

    def key_callback(self, keycode):
        """Pauses or unpauses the simulation when the space key is pressed."""
        if chr(keycode) == ' ':
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            self.get_logger().info(f"Simulation {status} at t={self.current_time:.3f}s")


def main(args=None):
    rclpy.init(args=args)
    
    #change the task here: 'regulation', 'circle', 'line', 'sphere'
    task = 'sphere'
    
    # Allow task to be specified via command line argument
    if args and len(args) > 0:
        task = args[0]
    
    controller_node = MujocoIKControllerNode(task=task)

    try:
        controller_node.MujocoSim()
    except KeyboardInterrupt:
        pass
    finally:
        controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    import sys
    # Pass command line arguments (skip the script name)
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
