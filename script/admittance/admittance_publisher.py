import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
import numpy as np
import sympy as sp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from misc_func import calculate_desired_pose_trajectory, vee_map, hat_map, adjoint_g_ed, adjoint_g_ed_dual, adjoint_g
from filter import ButterLowPass
from scipy.linalg import block_diag, expm

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
        
        # Manual discretization using matrix exponential
        # For system dx/dt = Ax + Bu, discrete form is x[k+1] = Ad*x[k] + Bd*u[k]
        # where Ad = exp(A*dt) and Bd = integral_0^dt(exp(A*tau)*B*dtau)
        
        # Calculate Ad = exp(A*dt)
        Ad1 = expm(A * dt)
        
        # Calculate Bd using approximation: Bd ≈ A^(-1)*(Ad - I)*B
        if np.linalg.det(A) != 0:
            Bd1 = np.linalg.inv(A) @ (Ad1 - np.eye(2)) @ B
        else:
            # If A is singular, use simple approximation Bd ≈ B*dt
            Bd1 = B * dt

        # stack Ad and Bd for dim times
        Ad = block_diag(*[Ad1 for _ in range(dim)])
        Bd = block_diag(*[Bd1 for _ in range(dim)])

        return Ad, Bd
    
    def lp_filter_implemented(self, force_torque):
        # 0, 2, 4, 6, 8, 10 indices are filtered values
        xf = self.filter_state[::2]

        # 1, 3, 5, 7, 9, 11 indices are filtered derivative values
        dxf = self.filter_state[1::2]

        self.filter_state = self.Ad @ self.filter_state + self.Bd @ force_torque.reshape((-1,1))

        return xf, dxf

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
        # Use the precise force sensor at the sensor body location
        try:
            # Use individual force components from the precise sensor
            fx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor_fx")
            fy_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor_fy") 
            fz_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor_fz")
            
            fx = self.data.sensordata[self.model.sensor_adr[fx_id]]
            fy = self.data.sensordata[self.model.sensor_adr[fy_id]]
            fz = self.data.sensordata[self.model.sensor_adr[fz_id]]
            force = np.array([fx, fy, fz])
            
            # Get torque components
            mx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor_mx")
            my_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor_my")
            mz_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor_mz")
            
            mx = self.data.sensordata[self.model.sensor_adr[mx_id]]
            my = self.data.sensordata[self.model.sensor_adr[my_id]]
            mz = self.data.sensordata[self.model.sensor_adr[mz_id]]
            torque = np.array([mx, my, mz])
            
        except:
            # Fallback to legacy sensors
            print("Warning: Using legacy force sensors at end effector position")
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_force_sensor")
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            force = np.copy(self.data.sensordata[adr:adr + dim])
            
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "jaka_torque_sensor")
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            torque = np.copy(self.data.sensordata[adr:adr + dim])

        force_torque = np.concatenate([force, torque])

        # Apply filtering
        ft, dft = self.lp_filter_implemented(force_torque)
        return ft, dft

class MujocoNode(Node):
    def __init__(self, task='sphere'):
        super().__init__('mujoco_ik_controller_node')
        self.n = 6
        self.xml_file = 'models/jaka_zu12/jaka_admittance.xml'
        self.task = task  # Task type: 'regulation', 'circle', 'line', 'sphere'
        
        # Initial joint configuration will be set based on task
        self.initial_q = None
        
        # Trajectory parameters
        # Set duration based on task type
        if task == 'circle':
            self.duration = 30.0  # 100 seconds for circle task
        else:
            self.duration = 10.0   # 10 seconds for other tasks
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

        # Admittance control parameters
        # Virtual mass, damping, and stiffness in Cartesian space (6x6 diagonal matrices)
        self.Mc = np.diag([20.0, 20.0, 20.0, 5, 5, 5])  # Virtual mass matrix
        self.Dc = np.diag([200.0, 200.0, 200.0, 50.0, 50.0, 50.0])  # Virtual damping matrix  
        self.Kc = np.diag([500.0, 500.0, 500.0, 200.0, 200.0, 200.0])  # Virtual stiffness matrix

        # Admittance states
        self.x_admittance = np.zeros(6)  # Admittance position/orientation displacement [px, py, pz, rx, ry, rz]
        self.dx_admittance = np.zeros(6)  # Admittance velocity
        self.ddx_admittance = np.zeros(6)  # Admittance acceleration

        # Trajectory recording for plotting
        self.trajectory_data = {
            'time': [],
            'desired_pos': [],
            'actual_pos': [],
            'modified_pos': [],
            'desired_rot': [],
            'actual_rot': [],
            'modified_rot': [],
            'force': [],
            'admittance_displacement': [],
            'joint_angles': []
        }

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

    def update_admittance_dynamics(self, Fe, dt):
        """
        Update admittance dynamics based on external force/torque.
        
        Args:
            Fe: External force/torque vector [fx, fy, fz, mx, my, mz]
            dt: Time step
            
        Returns:
            x_desired: Updated desired pose displacement
        """
        # Admittance control equation: Mc*(ddx) + Dc*(dx) + Kc*(x) = Fe
        # Rearranged: ddx = Mc_inv * (Fe - Dc*(dx) - Kc*(x))
        
        # Ensure Fe is a 1D array (flatten if needed)
        if Fe.ndim > 1:
            Fe = Fe.flatten()
        
        Mc_inv = np.linalg.inv(self.Mc)
        # M = self.robot_state.get_full_inertia()
        # Jb = self.robot_state.get_body_jacobian()
        # M_tilde_inv = Jb @ np.linalg.pinv(M) @ Jb.T
        # Calculate acceleration from admittance dynamics
        self.ddx_admittance = Mc_inv @ (Fe - self.Dc @ self.dx_admittance - self.Kc @ self.x_admittance)
        
        # Integrate to get velocity and position using simple Euler integration
        self.dx_admittance += self.ddx_admittance * dt
        self.x_admittance += self.dx_admittance * dt
        
        # Apply reasonable limits to prevent excessive displacement
        max_translation = 0.1  # 10cm max displacement
        max_rotation = 0.2     # ~11 degrees max rotation
        
        # Limit translation
        self.x_admittance[:3] = np.clip(self.x_admittance[:3], -max_translation, max_translation)
        # Limit rotation
        self.x_admittance[3:] = np.clip(self.x_admittance[3:], -max_rotation, max_rotation)
        
        return self.x_admittance.copy()

    def set_admittance_parameters(self, Mc=None, Dc=None, Kc=None):
        """
        Set admittance control parameters.
        
        Args:
            Mc: Virtual mass matrix (6x6) or diagonal values (6,)
            Dc: Virtual damping matrix (6x6) or diagonal values (6,)
            Kc: Virtual stiffness matrix (6x6) or diagonal values (6,)
        """
        if Mc is not None:
            if Mc.ndim == 1:
                self.Mc = np.diag(Mc)
            else:
                self.Mc = Mc
                
        if Dc is not None:
            if Dc.ndim == 1:
                self.Dc = np.diag(Dc)
            else:
                self.Dc = Dc
                
        if Kc is not None:
            if Kc.ndim == 1:
                self.Kc = np.diag(Kc)
            else:
                self.Kc = Kc

    def apply_admittance_to_desired_pose(self, pd_desired, Rd_desired, x_admittance):
        """
        Apply admittance displacement to the desired pose.
        
        Args:
            pd_desired: Original desired position [3x1]
            Rd_desired: Original desired rotation matrix [3x3]
            x_admittance: Admittance displacement [px, py, pz, rx, ry, rz]
            
        Returns:
            pd_modified: Modified desired position
            Rd_modified: Modified desired rotation matrix
        """
        # Apply translational displacement
        pd_modified = pd_desired + x_admittance[:3]
        
        # Apply rotational displacement using axis-angle representation
        rotation_displacement = x_admittance[3:]
        angle = np.linalg.norm(rotation_displacement)
        
        if angle > 1e-6:  # Avoid division by zero
            axis = rotation_displacement / angle
            # Create rotation matrix from axis-angle
            R_displacement = expm(hat_map(rotation_displacement))
            # Apply rotation displacement to desired orientation
            Rd_modified = R_displacement @ Rd_desired
        else:
            Rd_modified = Rd_desired.copy()
            
        return pd_modified, Rd_modified

    def MujocoSim(self):
        """
        Runs the MuJoCo simulation with real-time IK trajectory generation and control.
        """
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)
        self.robot_state = RobotState(model, data, "jaka_end_effector", "jaka")

        # Initialize IK solver with higher precision
        # Since we have accurate initial position, we can use stricter tolerance
        self.ik_solver = IKArm(solver_type='QP', tol=1.0e-6, ilimit=10000)
        
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
                        
                        # Get filtered external force/torque
                        Fe, dFe = self.robot_state.get_ee_force()
                        # Fe = np.array([0,0,1,0,0,0])
                        
                        # Update admittance dynamics to get pose modification
                        x_admittance = self.update_admittance_dynamics(Fe, model.opt.timestep)
                        
                        # Apply admittance displacement to desired pose
                        pd_modified, Rd_modified = self.apply_admittance_to_desired_pose(
                            pd_current, Rd_current, x_admittance
                        )
                        
                        # Construct the target pose matrix (Tep) with admittance-modified pose
                        Tep = np.eye(4)
                        Tep[:3, :3] = Rd_modified
                        Tep[:3, 3] = pd_modified
                        
                        # Solve IK for desired joint angles
                        q_sol, success, iterations, error, jl_valid, solve_time = self.ik_solver.solve(
                            model, data, Tep, self.previous_q
                        )
                        
                        if success:
                            # Apply position filtering for smoother control
                            alpha = 0.5  # position filter coefficient
                            self.desired_q = alpha * q_sol + (1 - alpha) * self.desired_q

                            # Set control commands
                            data.ctrl[:] = self.desired_q
                            self.previous_q = q_sol.copy()
                            
                            # Record trajectory data for plotting
                            current_pos, current_rot = self.robot_state.get_pose()
                            self.trajectory_data['time'].append(self.current_time)
                            self.trajectory_data['desired_pos'].append(pd_current.copy())
                            self.trajectory_data['actual_pos'].append(current_pos.copy())
                            self.trajectory_data['modified_pos'].append(pd_modified.copy())
                            self.trajectory_data['desired_rot'].append(Rd_current.copy())
                            self.trajectory_data['actual_rot'].append(current_rot.copy())
                            self.trajectory_data['modified_rot'].append(Rd_modified.copy())
                            self.trajectory_data['force'].append(Fe.flatten().copy())
                            self.trajectory_data['admittance_displacement'].append(x_admittance.copy())
                            self.trajectory_data['joint_angles'].append(data.qpos[:self.n].copy())
                            
                            # Optional: Log admittance state for debugging
                            if self.current_time % 1.0 < self.control_timestep:  # Log every second
                                self.get_logger().info(
                                    f"t={self.current_time:.2f}s, Fe_norm={np.linalg.norm(Fe):.3f}, "
                                    f"x_adm=[{self.x_admittance[0]:.3f}, {self.x_admittance[1]:.3f}, {self.x_admittance[2]:.3f}]"
                                )
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
            
            # Save trajectory data to file
            self.save_trajectory_data()

    def save_trajectory_data(self):
        """Save recorded trajectory data to files for plotting."""
        import pickle
        import csv
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'log')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save as pickle for easy loading in Python
        pickle_file = os.path.join(data_dir, f'admittance_trajectory_{self.task}.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.trajectory_data, f)
        
        # Save key data as CSV for other analysis tools
        csv_file = os.path.join(data_dir, f'admittance_trajectory_{self.task}.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['time', 
                     'desired_x', 'desired_y', 'desired_z',
                     'actual_x', 'actual_y', 'actual_z',
                     'modified_x', 'modified_y', 'modified_z',
                     'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z',
                     'adm_disp_x', 'adm_disp_y', 'adm_disp_z', 'adm_disp_rx', 'adm_disp_ry', 'adm_disp_rz']
            header.extend([f'joint_{i+1}' for i in range(self.n)])
            writer.writerow(header)
            
            # Write data
            for i in range(len(self.trajectory_data['time'])):
                row = [self.trajectory_data['time'][i]]
                row.extend(self.trajectory_data['desired_pos'][i])
                row.extend(self.trajectory_data['actual_pos'][i])
                row.extend(self.trajectory_data['modified_pos'][i])
                row.extend(self.trajectory_data['force'][i])
                row.extend(self.trajectory_data['admittance_displacement'][i])
                row.extend(self.trajectory_data['joint_angles'][i])
                writer.writerow(row)
        
        self.get_logger().info(f"Trajectory data saved to:")
        self.get_logger().info(f"  - {pickle_file}")
        self.get_logger().info(f"  - {csv_file}")

    def key_callback(self, keycode):
        """Pauses or unpauses the simulation when the space key is pressed."""
        if chr(keycode) == ' ':
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            self.get_logger().info(f"Simulation {status} at t={self.current_time:.3f}s")


def main(args=None):
    rclpy.init(args=args)
    
    #change the task here: 'regulation', 'circle', 'line', 'sphere'
    # task = input("Enter task ('regulation', 'circle', 'line', 'sphere'):")
    task = "circle"
    
    # Allow task to be specified via command line argument
    if args and len(args) > 0:
        task = args[0]
    
    controller_node = MujocoNode(task = task)
    
    # Optional: Customize admittance parameters
    # Lower mass -> more responsive to forces
    # Higher damping -> more stable but slower response  
    # Higher stiffness -> stronger return to original trajectory
    # controller_node.set_admittance_parameters(
    #     Mc=np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2]),  # Virtual mass
    #     Dc=np.array([15.0, 15.0, 15.0, 3.0, 3.0, 3.0]),  # Virtual damping
    #     Kc=np.array([30.0, 30.0, 30.0, 5.0, 5.0, 5.0])   # Virtual stiffness
    # )

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
