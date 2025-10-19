import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
import numpy as np
import sympy as sp
import sys
import os

# Add project root to Python path to allow importing from 'controllers'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
    def __init__(self):
        super().__init__('mujoco_ik_controller_node')
        self.n = 6
        self.xml_file = 'models/jaka_zu12/jaka_admittance.xml'
        
        # Initial joint configuration - computed via IK to match trajectory starting point
        self.initial_q = np.array([-1.9983232592, 1.2275963802, -2.0260540306, 2.1428747691, 2.0423938398, 1.0797850903])
        
        # Trajectory parameters
        self.duration = 10.0  # Total trajectory duration in seconds
        self.current_time = 0.0
        self.control_timestep = 0.008  # Update control every 12.5ms (125Hz)
        self.last_control_time = 0.0
        
        # Paused state
        self.paused = False
        
        # Initialize trajectory functions
        self.pd_t, self.Rd_t = self.calculate_desired_pose_trajectory()
        
        # Initialize IK solver
        self.ik_solver = None
        
        # Previous joint angles for warm start
        self.previous_q = self.initial_q.copy()
        
        self.get_logger().info("MuJoCo IK Controller Node initialized")

    def calculate_desired_pose_trajectory(self):
        """
        Defines the symbolic trajectory for the end-effector.
        """
        t = sp.symbols('t')
        max_time = 10
        total_radian = np.pi / 3
        omega_value = total_radian / max_time
        theta = omega_value * t - total_radian * 0.5
        r_sphere = 0.110

        pd_default = np.array([0.0, -0.7, -0.028])  # Center of the sphere
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        
        pd_default_sym = sp.Matrix([float(x) for x in pd_default])
        Rd_default_sym = sp.Matrix([[float(x) for x in row] for row in Rd_default])
        
        pd_t_sim = pd_default_sym + sp.Matrix([r_sphere * sp.sin(theta), 0, r_sphere * sp.cos(theta)])
        rotmat_y = sp.Matrix([[1, 0, 0], [0, sp.cos(theta), -sp.sin(theta)], [0, sp.sin(theta), sp.cos(theta)]])
        Rd_t_sim = Rd_default_sym @ rotmat_y

        pd_t = sp.lambdify(t, pd_t_sim, "numpy")
        Rd_t = sp.lambdify(t, Rd_t_sim, "numpy")

        return pd_t, Rd_t

    def MujocoSim(self):
        """
        Runs the MuJoCo simulation with real-time IK trajectory generation and control.
        """
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)
        
        # Initialize IK solver with higher precision
        # Since we have accurate initial position, we can use stricter tolerance
        self.ik_solver = IKArm(solver_type='QP', tol=1e-5, ilimit=200)
        
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
                            # Set control commands to desired joint angles
                            data.ctrl[:self.n] = q_sol
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
    controller_node = MujocoIKControllerNode()

    try:
        controller_node.MujocoSim()
    except KeyboardInterrupt:
        pass
    finally:
        controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
