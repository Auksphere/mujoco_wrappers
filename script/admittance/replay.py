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
        self.trajectory_csv_file = 'log/ik_trajectory.csv'
        
        self.paused = False
        self.trajectory = self.load_trajectory()

    def load_trajectory(self):
        """Loads the joint trajectory from a CSV file."""
        trajectory = []
        with open(self.trajectory_csv_file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader) # Skip header
            for row in csv_reader:
                # row[0] is time, row[1:] are joint angles
                trajectory.append([float(val) for val in row[1:]])
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
        
        traj_index = 0
        
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

                # Maintain simulation frequency
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                rclpy.spin_once(self, timeout_sec=0.0)
            
            self.get_logger().info("Trajectory playback finished.")

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