#!/usr/bin/env python3
"""
Simplified trajectory plotting program for admittance control analysis.
This script loads recorded trajectory data and creates two main plots:
1. 3D trajectory comparison (desired vs actual vs admittance-modified)
2. End-effector force analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import sys
import argparse

class TrajectoryPlotter:
    def __init__(self, data_file=None, task='circle'):
        """
        Initialize trajectory plotter.
        
        Args:
            data_file: Path to pickle file containing trajectory data
            task: Task name if loading from default location
        """
        self.task = task
        self.data = None
        
        if data_file is None:
            # Default data file location
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'log')
            data_file = os.path.join(data_dir, f'admittance_trajectory_{task}.pkl')
        
        self.data_file = data_file
        self.load_data()
        
    def load_data(self):
        """Load trajectory data from pickle file."""
        try:
            with open(self.data_file, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded trajectory data from: {self.data_file}")
            print(f"Data points: {len(self.data['time'])}")
            print(f"Duration: {self.data['time'][-1]:.2f} seconds")
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_file}")
            print("Please run the admittance controller first to generate data.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def plot_3d_trajectory_and_forces(self):
        """Plot 3D trajectory comparison and force analysis in one figure."""
        fig = plt.figure(figsize=(16, 8))
        
        # Convert lists to numpy arrays for easier manipulation
        time = np.array(self.data['time'])
        desired_pos = np.array(self.data['desired_pos'])
        actual_pos = np.array(self.data['actual_pos'])
        modified_pos = np.array(self.data['modified_pos'])
        force = np.array(self.data['force'])
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(desired_pos[:, 0], desired_pos[:, 1], desired_pos[:, 2], 
                'b-', linewidth=3, label='Desired trajectory', alpha=0.8)
        ax1.plot(modified_pos[:, 0], modified_pos[:, 1], modified_pos[:, 2], 
                'g--', linewidth=2, label='Admittance-modified', alpha=0.8)
        ax1.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], 
                'r:', linewidth=2, label='Actual trajectory', alpha=0.8)
        
        # Mark start and end points
        ax1.scatter(desired_pos[0, 0], desired_pos[0, 1], desired_pos[0, 2], 
                   color='green', s=100, label='Start', marker='o')
        ax1.scatter(desired_pos[-1, 0], desired_pos[-1, 1], desired_pos[-1, 2], 
                   color='red', s=100, label='End', marker='s')
        
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.set_title(f'3D Trajectory ({self.task.capitalize()})')
        ax1.legend()
        ax1.grid(True)
        
        # X-Y plane view
        ax2 = fig.add_subplot(232)
        ax2.plot(desired_pos[:, 0], desired_pos[:, 1], 'b-', linewidth=3, 
                label='Desired', alpha=0.8)
        ax2.plot(modified_pos[:, 0], modified_pos[:, 1], 'g--', linewidth=2, 
                label='Admittance-modified', alpha=0.8)
        ax2.plot(actual_pos[:, 0], actual_pos[:, 1], 'r:', linewidth=2, 
                label='Actual', alpha=0.8)
        ax2.scatter(desired_pos[0, 0], desired_pos[0, 1], color='green', s=100, marker='o')
        ax2.scatter(desired_pos[-1, 0], desired_pos[-1, 1], color='red', s=100, marker='s')
        
        # Add circle center for circle task
        if self.task == 'circle':
            center_x = np.mean(desired_pos[:, 0])
            center_y = np.mean(desired_pos[:, 1])
            radius = np.mean(np.sqrt((desired_pos[:, 0] - center_x)**2 + (desired_pos[:, 1] - center_y)**2))
            ax2.scatter(center_x, center_y, color='black', s=50, marker='+', 
                       label=f'Center (R={radius:.3f}m)')
        
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_title('X-Y Plane View')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # Force components over time
        ax3 = fig.add_subplot(233)
        ax3.plot(time, force[:, 0], 'r-', linewidth=2, label='Fx')
        ax3.plot(time, force[:, 1], 'g-', linewidth=2, label='Fy')
        ax3.plot(time, force[:, 2], 'b-', linewidth=2, label='Fz')
        force_magnitude = np.linalg.norm(force[:, :3], axis=1)
        ax3.plot(time, force_magnitude, 'k--', linewidth=2, label='|F|')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Force [N]')
        ax3.set_title('End-Effector Forces')
        ax3.legend()
        ax3.grid(True)
        
        # Torque components over time
        ax4 = fig.add_subplot(234)
        ax4.plot(time, force[:, 3], 'r--', linewidth=2, label='Mx')
        ax4.plot(time, force[:, 4], 'g--', linewidth=2, label='My')
        ax4.plot(time, force[:, 5], 'b--', linewidth=2, label='Mz')
        torque_magnitude = np.linalg.norm(force[:, 3:6], axis=1)
        ax4.plot(time, torque_magnitude, 'k:', linewidth=2, label='|M|')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Torque [Nm]')
        ax4.set_title('End-Effector Torques')
        ax4.legend()
        ax4.grid(True)
        
        # Position tracking error
        ax5 = fig.add_subplot(235)
        error_desired = np.linalg.norm(actual_pos - desired_pos, axis=1)
        error_modified = np.linalg.norm(actual_pos - modified_pos, axis=1)
        ax5.plot(time, error_desired * 1000, 'b-', linewidth=2, label='vs Desired')
        ax5.plot(time, error_modified * 1000, 'r-', linewidth=2, label='vs Modified')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Position Error [mm]')
        ax5.set_title('Position Tracking Error')
        ax5.legend()
        ax5.grid(True)
        
        # Admittance displacement
        ax6 = fig.add_subplot(236)
        admittance_disp = np.array(self.data['admittance_displacement'])
        ax6.plot(time, admittance_disp[:, 0] * 1000, 'r-', linewidth=2, label='dx')
        ax6.plot(time, admittance_disp[:, 1] * 1000, 'g-', linewidth=2, label='dy')
        ax6.plot(time, admittance_disp[:, 2] * 1000, 'b-', linewidth=2, label='dz')
        disp_magnitude = np.linalg.norm(admittance_disp[:, :3], axis=1)
        ax6.plot(time, disp_magnitude * 1000, 'k--', linewidth=2, label='|disp|')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Displacement [mm]')
        ax6.set_title('Admittance Displacement')
        ax6.legend()
        ax6.grid(True)
        
        plt.suptitle(f'Admittance Control Analysis - {self.task.capitalize()} Task', fontsize=16)
        plt.tight_layout()
        
        # Print statistics
        print("\n=== TRAJECTORY ANALYSIS ===")
        print(f"Task: {self.task}")
        print(f"Duration: {time[-1]:.2f} seconds")
        print(f"Average force magnitude: {np.mean(force_magnitude):.3f} N")
        print(f"Max force magnitude: {np.max(force_magnitude):.3f} N")
        print(f"RMS tracking error (vs desired): {np.sqrt(np.mean(error_desired**2)) * 1000:.2f} mm")
        print(f"RMS tracking error (vs modified): {np.sqrt(np.mean(error_modified**2)) * 1000:.2f} mm")
        print(f"Max admittance displacement: {np.max(disp_magnitude) * 1000:.2f} mm")
        
        return fig
    
    def save_plots(self, output_dir=None):
        """Generate and save simplified plots."""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'plots')
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating simplified plots for task: {self.task}")
        
        # Generate main plot
        fig = self.plot_3d_trajectory_and_forces()
        
        # Save plot
        filename = os.path.join(output_dir, f'admittance_analysis_{self.task}.png')
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        # Show plot
        plt.show()
        
        return fig

def main():
    parser = argparse.ArgumentParser(description='Plot admittance control trajectory data')
    parser.add_argument('--task', type=str, default='circle', 
                       choices=['regulation', 'circle', 'line', 'sphere'],
                       help='Task type to plot')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to trajectory data file (overrides --task)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = TrajectoryPlotter(data_file=args.data_file, task=args.task)
    
    # Generate and save simplified plots
    plotter.save_plots(output_dir=args.output_dir)

if __name__ == '__main__':
    main()
