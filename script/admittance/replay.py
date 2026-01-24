import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
import numpy as np
import csv

class MujocoPlaybackNode(Node):
    def __init__(self):
        super().__init__('mujoco_playback_node')
        self.n = 6
        self.xml_file = 'models/jaka_zu12/jaka_admittance.xml'

        self.trajectory_csv_file = 'log/trajectory_sphere.csv'
        
        self.paused = False
        self.trajectory = self.load_trajectory()

    def load_trajectory(self):
        """Loads the joint trajectory from a CSV file."""
        trajectory = []
        with open(self.trajectory_csv_file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)  # Skip header and get column names
            
            # Find joint column indices (joint_1 to joint_6)
            joint_indices = []
            for i, col_name in enumerate(header):
                if col_name.startswith('joint_'):
                    joint_indices.append(i)
            
            self.get_logger().info(f"Found joint columns at indices: {joint_indices}")
            
            for row in csv_reader:
                # Extract only joint angles (last 6 columns)
                joint_angles = [float(row[i]) for i in joint_indices]
                trajectory.append(joint_angles)
                
        self.get_logger().info(f"Loaded {len(trajectory)} points from {self.trajectory_csv_file}")
        return trajectory

    def MujocoSim(self):
        """Runs the MuJoCo simulation, playing back the pre-computed trajectory."""
        if not self.trajectory:
            self.get_logger().error("Trajectory is empty. Cannot start simulation.")
            return

        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)

        # Set initial joint positions from the first point in the trajectory
        initial_qpos = self.trajectory[0]
        data.qpos[:self.n] = initial_qpos
        mujoco.mj_forward(model, data)  # Forward kinematics to update positions
        
        traj_index = 0
        start_time = time.time()
        
        self.get_logger().info(f"Starting trajectory playback with {len(self.trajectory)} points")
        self.get_logger().info("Controls: Space = Pause/Resume")
        
        with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
            while viewer.is_running() and traj_index < len(self.trajectory):
                step_start = time.time()

                if not self.paused:
                    # Get the desired position for the current step
                    desired_position = self.trajectory[traj_index]
                    data.ctrl[:self.n] = desired_position
                    
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    traj_index += 1
                    
                    # Progress reporting every 100 steps
                    if traj_index % 100 == 0:
                        progress = (traj_index / len(self.trajectory)) * 100
                        elapsed = time.time() - start_time
                        self.get_logger().info(f"Replay progress: {progress:.1f}% ({traj_index}/{len(self.trajectory)})")

                # Maintain simulation frequency
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                rclpy.spin_once(self, timeout_sec=0.0)
            

    def key_callback(self, keycode):
        """Pauses or unpauses the simulation when the space key is pressed."""
        if chr(keycode) == ' ':
            self.paused = not self.paused

def main(args=None):
    rclpy.init(args=args)
    playback_node = MujocoPlaybackNode()

    try:
        playback_node.MujocoSim()
    except KeyboardInterrupt:
        pass
    finally:
        playback_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()