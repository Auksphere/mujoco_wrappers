import mujoco
import numpy as np
import sympy as sp
import csv
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

class TrajectoryGenerator:
    def __init__(self):
        self.n = 6
        self.xml_file = 'models/jaka_zu12/jaka_admittance.xml'
        self.output_csv_file = 'log/ik_trajectory.csv'
        
        # Initial joint configuration for the first IK solve
        self.initial_q = np.array([-1.95, 1.24, -2.04, 2.22, 1.94, 1.16])
        
        # Initialize trajectory functions
        self.pd_t, self.Rd_t = self.calculate_desired_pose_trajectory()

    def calculate_desired_pose_trajectory(self):
        """
        Defines the symbolic trajectory for the end-effector.
        """
        t = sp.symbols('t')
        max_time = 10
        total_radian = np.pi / 3
        omega_value = total_radian / max_time
        theta = omega_value * t - total_radian * 0.5
        r_sphere = 0.100

        pd_default = np.array([0.0, -0.7, -0.028]) # Center of the sphere
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        
        pd_default_sym = sp.Matrix([float(x) for x in pd_default])
        Rd_default_sym = sp.Matrix([[float(x) for x in row] for row in Rd_default])
        
        pd_t_sim = pd_default_sym + sp.Matrix([r_sphere * sp.sin(theta), 0, r_sphere * sp.cos(theta)])
        rotmat_y = sp.Matrix([[1, 0, 0], [0, sp.cos(theta), -sp.sin(theta)], [0, sp.sin(theta), sp.cos(theta)]])
        Rd_t_sim = Rd_default_sym @ rotmat_y

        pd_t = sp.lambdify(t, pd_t_sim, "numpy")
        Rd_t = sp.lambdify(t, Rd_t_sim, "numpy")

        return pd_t, Rd_t

    def generate(self, duration=10.0, timestep=0.002):
        """
        Generates the joint trajectory using IK and saves it to a CSV file.
        """
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)
        
        # Use the IKArm solver from the controllers package
        ik_solver = IKArm(solver_type='QP', tol=5e-3, ilimit=100)
        
        previous_q = self.initial_q.copy()
        
        print(f"Generating trajectory for {duration} seconds...")
        with open(self.output_csv_file, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            header = ['time'] + [f'q{i+1}' for i in range(self.n)]
            csv_writer.writerow(header)

            num_steps = int(duration / timestep)
            for i in range(num_steps):
                current_time = i * timestep
                
                # Calculate desired end-effector pose
                pd_current = np.array(self.pd_t(current_time)).flatten()
                Rd_current = np.array(self.Rd_t(current_time)).reshape(3, 3)
                
                # Construct the target pose matrix (Tep)
                Tep = np.eye(4)
                Tep[:3, :3] = Rd_current
                Tep[:3, 3] = pd_current
                
                # Use previous solution as initial guess (warm start)
                init_q = previous_q
                
                # Solve IK
                q_sol, success, _, _, _, _ = ik_solver.solve(model, data, Tep, init_q)
                
                if success:
                    csv_writer.writerow([current_time] + list(q_sol))
                    previous_q = q_sol.copy()
                else:
                    print(f"Warning: IK solution failed at time {current_time:.3f}s. Reusing previous solution.")
                    # Write the previous successful solution to maintain trajectory continuity
                    csv_writer.writerow([current_time] + list(previous_q))

        print(f"Trajectory saved to {self.output_csv_file}")

def main():
    generator = TrajectoryGenerator()
    generator.generate()

if __name__ == '__main__':
    main()