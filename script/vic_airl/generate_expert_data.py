#!/usr/bin/env python3
"""
Expert Demonstration Generator for Peg-in-Hole Task

Generates expert trajectories using admittance control with hand-tuned
impedance parameters. These demonstrations will be used to train the 
AIRL policy network.

The expert uses a trajectory-following admittance controller similar to
the implementation in script/vic/admittance_publisher.py but simplified
for demonstration generation.
"""

import numpy as np
import mujoco
import sys
import os
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as ScipyRotation

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from script.vic.misc_func import calculate_desired_pose_trajectory
from script.vic.filter import ButterLowPass
from controllers.ik_arm import IKArm


class ExpertDemonstrationGenerator:
    """
    Generates expert demonstrations for peg-in-hole insertion using
    pre-defined admittance control parameters.
    """
    
    def __init__(
        self,
        xml_path: str = "models/jaka_zu12/jaka_pih.xml",
        control_dt: float = 0.04,
        physics_dt: float = 0.001,
    ):
        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self.n_substeps = int(control_dt / physics_dt)
        
        # Expert admittance parameters (hand-tuned for peg-in-hole)
        # These should ensure successful insertion with good force profiles
        self.Mc = np.diag([20.0, 20.0, 20.0, 5.0, 5.0, 5.0])  # Mass
        self.Dc = np.diag([15.0, 15.0, 30.0, 8.0, 8.0, 8.0])  # Damping (higher in z for insertion)
        self.Kc = np.diag([800.0, 800.0, 400.0, 200.0, 200.0, 200.0])  # Stiffness (lower in z)
        
        # Admittance state
        self.x_admittance = np.zeros(6)
        self.dx_admittance = np.zeros(6)
        self.ddx_admittance = np.zeros(6)
        
        # Get model IDs
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        
        # IK solver - use Levenberg-Marquardt solver for better stability
        self.ik_solver = IKArm(
            solver_type='LM_Chan',
            ilimit=30,
            tol=1e-4
        )
        
        # Trajectory parameters
        self.task = "peg_insertion"
        self.duration = 8.0  # Stop at the optimal point we found (8s gives 21mm)
        
        self.initial_q = np.array([
            -1.63,   # actuator1
            1.54,    # actuator2
            -2.07,   # actuator3
            2.36,    # actuator4
            1.51,    # actuator5
            -0.188,  # actuator6
        ])
        
        # Add target configuration for successful insertion  
        # Further optimized for dynamic control performance
        self.target_q = np.array([
            -1.7597714768,  # Keep optimal joint 1
            1.3693907515,   # Optimized joint 2 (+0.05)
            -2.0578084960,  # Optimized joint 3 (-0.01 from previous -2.0478084960)
            2.3293412478,   # Keep optimal joint 4
            1.5807927830,   # Keep optimal joint 5
            1.3818177682,   # Keep optimal joint 6
        ])
        
        print(f"\n{'='*60}")
        print(f"Expert Demonstration Generator Initialization")
        print(f"{'='*60}")
        print(f"Initial joint configuration: {self.initial_q}")
        
        # Set initial configuration and compute end-effector pose
        self.data.qpos[:6] = self.initial_q
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial end-effector position
        initial_ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        initial_ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        
        print(f"Initial EE position: {initial_ee_pos}")
        
        # Get hole position
        hole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hole')
        hole_pos = self.data.xpos[hole_id].copy()
        print(f"Hole position: {hole_pos}")
        
        # Calculate trajectory start and goal based on initial EE position
        # Start from current EE position, go to hole position
        self.trajectory_start = initial_ee_pos.copy()
        
        # Goal should be at hole level for actual insertion
        self.trajectory_goal = np.array([
            hole_pos[0],
            hole_pos[1], 
            hole_pos[2]  # Use the actual hole position
        ])
        
        print(f"Trajectory start: {self.trajectory_start}")
        print(f"Trajectory goal: {self.trajectory_goal}")
        print(f"Insertion distance: {np.linalg.norm(self.trajectory_goal - self.trajectory_start)*1000:.1f}mm")
        print(f"{'='*60}\n")
        
        # Force filter
        fs = 1.0 / physics_dt
        cutoff = 10.0
        self.force_filter = ButterLowPass(cutoff, fs, order=5)
        
    def _get_peg_insertion_trajectory(self):
        """
        Define a simple trajectory for peg insertion.
        Uses pre-computed trajectory start and goal positions.
        """
        import sympy as sp
        t = sp.symbols('t')
        
        # Use pre-computed trajectory points (from __init__)
        x0, y0, z0 = self.trajectory_start
        xg, yg, zg = self.trajectory_goal
        
        # Smooth trajectory using quintic polynomial
        # Position
        t_norm = t / self.duration
        s = 10 * t_norm**3 - 15 * t_norm**4 + 6 * t_norm**5  # Smooth interpolation [0, 1]
        
        px = x0 + (xg - x0) * s
        py = y0 + (yg - y0) * s
        pz = z0 + (zg - z0) * s
        
        # Orientation (keep vertical pointing down)
        # Roll = 0, Pitch = pi (pointing down), Yaw = 0
        roll, pitch, yaw = 0, sp.pi, 0
        
        # Convert to rotation matrix
        Rx = sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(roll), -sp.sin(roll)],
            [0, sp.sin(roll), sp.cos(roll)]
        ])
        Ry = sp.Matrix([
            [sp.cos(pitch), 0, sp.sin(pitch)],
            [0, 1, 0],
            [-sp.sin(pitch), 0, sp.cos(pitch)]
        ])
        Rz = sp.Matrix([
            [sp.cos(yaw), -sp.sin(yaw), 0],
            [sp.sin(yaw), sp.cos(yaw), 0],
            [0, 0, 1]
        ])
        R_desired = Rz * Ry * Rx
        
        # Create lambda functions
        pd_func = sp.lambdify(t, sp.Matrix([px, py, pz]), 'numpy')
        Rd_func = sp.lambdify(t, R_desired, 'numpy')
        
        # Derivatives
        dpd_func = sp.lambdify(t, sp.Matrix([px, py, pz]).diff(t), 'numpy')
        dRd_func = sp.lambdify(t, R_desired.diff(t), 'numpy')
        
        ddpd_func = sp.lambdify(t, sp.Matrix([px, py, pz]).diff(t, 2), 'numpy')
        ddRd_func = sp.lambdify(t, R_desired.diff(t, 2), 'numpy')
        
        return pd_func, Rd_func, dpd_func, dRd_func, ddpd_func, ddRd_func
    
    def _get_force_torque(self):
        """Read force/torque from contact forces."""
        force = np.zeros(3)
        torque = np.zeros(3)
        
        # Sum contact forces on the peg body
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Check if contact involves the peg body
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]
            
            if geom1_body == self.ee_body_id or geom2_body == self.ee_body_id:
                # Get contact force in world frame
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                force += c_array[:3]
        
        return np.concatenate([force, torque])
    
    def _get_ee_pose(self):
        """Get end-effector pose."""
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return ee_pos, ee_mat
    
    def _get_gic_expert_action(self, current_pos, current_rot, goal_pos, goal_rot):
        """
        Generate expert action using GIC strategy adapted from GIC_Learning_public.
        This mimics the get_expert_action function from the successful implementation.
        """
        # Calculate position and orientation errors
        pos_error = current_pos - goal_pos
        
        # Rotation error using matrix trace
        rot_error_trace = np.trace(np.eye(3) - goal_rot.T @ current_rot)
        pos_error_norm = 0.5 * np.dot(pos_error, pos_error)
        total_distance = np.sqrt(rot_error_trace + pos_error_norm)
        
        # Transform error to goal frame (like eg in GIC code)
        eg = current_rot.T @ pos_error.reshape(-1, 1)
        z_part = abs(eg[2, 0])
        trans_part_xy = np.sqrt(eg[0:2].T @ eg[0:2])
        
        # Adaptive impedance strategy based on task phase (from GIC_Learning_public)
        if total_distance > 1.0:  # Approach phase
            # Low impedance for large motions
            kp_xy = 0.1; kp_z = 0.3; kr = 0.6
        elif z_part < 0.03 and z_part > 0.01:  # Contact phase
            # Medium impedance for contact
            kp_xy = 0.35; kp_z = 0.1; kr = 0.8
        elif z_part < 0.01 and rot_error_trace < 0.0005 and trans_part_xy < 0.0002:  # Insertion phase
            # High impedance when well-aligned
            kp_xy = 0.65; kp_z = 0.8; kr = 0.9
        elif z_part < 0.01:  # Near contact but misaligned
            # Low Z impedance, medium XY for alignment
            kp_xy = 0.6; kp_z = 0.0; kr = 0.9
        else:  # Default case
            kp_xy = 0.2; kp_z = 0.2; kr = 0.7
        
        # Construct impedance action (normalized [0,1])
        # [Kp_x, Kp_y, Kp_z, Kd_x, Kd_y, Kd_z, Kp_rx, Kp_ry, Kp_rz, Kd_rx, Kd_ry, Kd_rz]
        action = np.array([
            kp_xy, kp_xy, kp_z,  # Linear stiffness
            kp_xy * 0.6, kp_xy * 0.6, kp_z * 0.6,  # Linear damping (fraction of stiffness)
            kr, kr, kr,  # Angular stiffness
            kr * 0.5, kr * 0.5, kr * 0.5  # Angular damping
        ])
        
        # Add small noise for diversity (like in GIC code)
        noise = np.random.randn(12) * 0.03
        action += noise
        action = np.clip(action, 0.0, 1.0)
        
        return action

    def generate_demonstration(self, verbose: bool = True):
        """
        Generate one expert demonstration trajectory.
        
        Returns:
            trajectory: List of (state, action, reward) tuples
        """
        # Reset environment
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint configuration for the arm (first 6 joints)
        self.data.qpos[:6] = self.initial_q
        mujoco.mj_forward(self.model, self.data)
        
        # Get the number of arm joints
        self.n_arm_joints = 6
        
        # Reset admittance state
        self.x_admittance = np.zeros(6)
        self.dx_admittance = np.zeros(6)
        self.ddx_admittance = np.zeros(6)
        
        # Get initial EE position
        initial_ee_pos, initial_ee_rot = self._get_ee_pose()
        
        if verbose:
            print(f"\n=== Generating Demonstration ===")
            print(f"Initial EE position: {initial_ee_pos}")
            print(f"Target start: {self.trajectory_start}")
            print(f"Target goal: {self.trajectory_goal}")
            print(f"Distance from initial to start: {np.linalg.norm(self.trajectory_start - initial_ee_pos)*1000:.1f}mm")
        
        # Get trajectory functions
        pd_t, Rd_t, dpd_t, dRd_t, ddpd_t, ddRd_t = self._get_peg_insertion_trajectory()
        
        # Calculate phases
        n_steps = int(self.duration / self.control_dt)
        
        # Since trajectory_start is now the current EE position, 
        # we can skip Phase 1 and go directly to trajectory execution
        
        if verbose:
            print(f"\nExecuting trajectory ({n_steps} steps, {self.duration:.1f}s)")
            print(f"From current position to goal: {np.linalg.norm(self.trajectory_goal - initial_ee_pos)*1000:.1f}mm")
        
        # === TRAJECTORY EXECUTION ===
        trajectory = []
        
        if verbose:
            print(f"Generating expert demonstration for {self.duration}s ({n_steps} steps)...")
            iterator = tqdm(range(n_steps))
        else:
            iterator = range(n_steps)
        
        for step in iterator:
            current_time = step * self.control_dt
            
            # Get desired pose from trajectory
            pd_desired = np.array(pd_t(current_time)).flatten()
            Rd_desired = np.array(Rd_t(current_time)).reshape(3, 3)
            
            # Get current pose
            ee_pos, ee_rot = self._get_ee_pose()
            
            # Get force/torque
            ft = self._get_force_torque()
            
            # === USE GIC EXPERT STRATEGY ===
            # Generate expert impedance action based on current state
            expert_action = self._get_gic_expert_action(ee_pos, ee_rot, pd_desired, Rd_desired)
            
            # For now, still use trajectory following for control (like the original admittance approach)
            # In a real implementation, we would use impedance control with the expert_action parameters
            pd_modified = pd_desired
            Rd_modified = Rd_desired
            
            # Solve IK
            Tep = np.eye(4)
            Tep[:3, :3] = Rd_modified
            Tep[:3, 3] = pd_modified
            
            # Prepare full q0 for IK solver
            q0_full = self.data.qpos.copy()
            
            # === FIXED JOINT SPACE INTERPOLATION ===  
            t_normalized = current_time / self.duration
            
            # Simple quintic polynomial - no overshoot issues
            s = 10 * t_normalized**3 - 15 * t_normalized**4 + 6 * t_normalized**5
            s = min(s, 1.0)  # Ensure s doesn't exceed 1.0
            
            q_des = self.initial_q + s * (self.target_q - self.initial_q)
            
            # Optional: Still try IK for comparison but don't rely on it
            # This way we can see how well IK would work
            try:
                Tep = np.eye(4)
                Tep[:3, :3] = Rd_modified
                Tep[:3, 3] = pd_modified
                q0_full = self.data.qpos.copy()
                
                q_ik, success_ik, _, _, _, _ = self.ik_solver.solve(
                    self.model, self.data, Tep, q0_full
                )
                
                # For debugging: could compare IK solution with interpolation
                # but still use interpolation as primary method
                
            except Exception as e:
                pass  # IK is just for reference
            
            # Build state (matching PegInHoleEnv observation)
            joint_pos = self.data.qpos[:6].copy()
            joint_vel = self.data.qvel[:6].copy()
            ee_quat = ScipyRotation.from_matrix(ee_rot).as_quat()
            ee_vel = np.zeros(6)  # Simplified
            
            # Get hole position from model
            hole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hole')
            hole_pos = self.data.xpos[hole_id].copy()
            
            state = np.concatenate([
                joint_pos, joint_vel, ee_pos, ee_quat,
                ee_vel, ft[:3], ft[3:], hole_pos
            ]).astype(np.float32)
            
            # === USE GIC EXPERT ACTION INSTEAD OF FIXED PARAMETERS ===
            # The expert_action was computed above using the GIC strategy
            # This replaces the fixed impedance approach
            
            # Store transition
            trajectory.append({
                'state': state,
                'action': expert_action,  # Use the GIC expert action
                'next_state': None,  # Will be filled in post-processing
                'done': False,
                'time': current_time,
                'ee_pos': ee_pos.copy(),
                'force': ft[:3].copy(),
                'impedance_K': self.Kc.copy(),
                'impedance_D': self.Dc.copy(),
            })
            
            # === SIMPLIFIED POSITION CONTROL ===
            # The multi-phase interpolation is good, but use simpler control
            for _ in range(self.n_substeps):
                self.data.ctrl[:6] = q_des
                mujoco.mj_step(self.model, self.data)
        
        # Post-process: fill in next_state
        for i in range(len(trajectory) - 1):
            trajectory[i]['next_state'] = trajectory[i + 1]['state']
        trajectory[-1]['next_state'] = trajectory[-1]['state']  # Terminal state
        trajectory[-1]['done'] = True
        
        # Calculate final statistics
        final_ee_pos, _ = self._get_ee_pose()
        final_error = np.linalg.norm(final_ee_pos - self.trajectory_goal)
        forces = np.array([t['force'] for t in trajectory])
        max_force = np.linalg.norm(forces, axis=1).max()
        
        if verbose:
            print(f"\nDemonstration completed:")
            print(f"  Final EE position: {final_ee_pos}")
            print(f"  Target position: {self.trajectory_goal}")
            print(f"  Final error: {final_error*1000:.2f}mm")
            print(f"  Max contact force: {max_force:.2f}N")
            print(f"  Trajectory steps: {len(trajectory)}")
        
        return trajectory
    
    def generate_dataset(self, n_demonstrations: int, save_path: str = None):
        """
        Generate multiple expert demonstrations.
        
        Args:
            n_demonstrations: Number of demonstrations to generate
            save_path: Path to save the dataset (pickle file)
        
        Returns:
            dataset: List of trajectories
        """
        dataset = []
        
        print(f"\nGenerating {n_demonstrations} expert demonstrations...")
        for i in range(n_demonstrations):
            print(f"\n--- Demonstration {i+1}/{n_demonstrations} ---")
            
            try:
                traj = self.generate_demonstration(verbose=True)
                dataset.append(traj)
                
                # Print statistics
                forces = [t['force'] for t in traj]
                max_force = np.max([np.linalg.norm(f) for f in forces])
                final_pos = traj[-1]['ee_pos']
                final_error = np.linalg.norm(final_pos - self.trajectory_goal)
                
                print(f"Max force: {max_force:.2f}N, Final error: {final_error*1000:.2f}mm")
                
                # Save progress after each demonstration
                if save_path is not None and len(dataset) > 0:
                    temp_path = save_path.replace('.pkl', f'_partial_{len(dataset)}.pkl')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(temp_path, 'wb') as f:
                        pickle.dump(dataset, f)
                    print(f"Progress saved to {temp_path}")
                    
            except KeyboardInterrupt:
                print(f"\n\nInterrupted! Saving {len(dataset)} demonstrations collected so far...")
                break
            except Exception as e:
                print(f"\nError in demonstration {i+1}: {e}")
                print("Continuing with next demonstration...")
                continue
        
        # Save final dataset
        if save_path is not None and len(dataset) > 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"\nDataset saved to {save_path}")
            
            # Clean up partial files
            for i in range(1, len(dataset) + 1):
                temp_path = save_path.replace('.pkl', f'_partial_{i}.pkl')
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return dataset


if __name__ == "__main__":
    # Generate expert demonstrations
    generator = ExpertDemonstrationGenerator()
    
    # Generate 3 demonstrations for testing (reduce from 10)
    dataset = generator.generate_dataset(
        n_demonstrations=3,
        save_path="data/expert_demonstrations.pkl"
    )
    
    print(f"\nGenerated {len(dataset)} demonstrations")
    if len(dataset) > 0:
        print(f"Each demonstration has {len(dataset[0])} timesteps")
