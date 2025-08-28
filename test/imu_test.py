
#!/usr/bin/env python3
"""Simple IMU -> MuJoCo cube visualizer (ROS2).

Subscriptions (high-rate):
 - /imu/data (sensor_msgs/Imu) preferred (contains orientation)
 - /filter/quaternion (geometry_msgs/QuaternionStamped) fallback

MuJoCo model: models/reorientation_cube/reorientation_cube.xml
Simulation rate: 400 Hz (dt = 0.0025)
"""

import os
import sys
import time
import threading
from typing import Optional

import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import QuaternionStamped

# Path to the MuJoCo model (cube)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'reorientation_cube', 'reorientation_cube.xml')


class IMUCubeVisualizer(Node):
	def __init__(self):
		super().__init__('imu_cube_visualizer')

		# state shared between ROS callbacks and viz thread
		self.lock = threading.Lock()
		# quaternion stored as [w, x, y, z]
		self.latest_quat: Optional[np.ndarray] = None

		# QoS tuned for high-rate IMU (use BEST_EFFORT for lower latency)
		high_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

		# prefer sensor_msgs/Imu topic (contains full orientation)
		self.create_subscription(Imu, '/imu/data', self.cb_imu, high_qos)
		# fallback/auxiliary quaternion topic
		self.create_subscription(QuaternionStamped, '/filter/quaternion', self.cb_quatstamped, high_qos)

		# Load MuJoCo model
		if not os.path.exists(MODEL_PATH):
			self.get_logger().error(f'MuJoCo model not found: {MODEL_PATH}')
			raise FileNotFoundError(MODEL_PATH)

		self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
		self.data = mujoco.MjData(self.model)

		# visualization viewer (passive mode)
		self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

		# control flags
		self.running = True

		# start viz loop thread running at 400Hz
		self.dt = 0.0025
		self.viz_thread = threading.Thread(target=self.viz_loop, daemon=True)
		self.viz_thread.start()

		self.get_logger().info('IMU cube visualizer initialized (400Hz)')

	# ROS callbacks
	def cb_imu(self, msg: Imu):
		# sensor_msgs/Imu orientation is (x,y,z,w)
		q = msg.orientation
		with self.lock:
			try:
				self.latest_quat = np.array([q.w, q.x, q.y, q.z], dtype=float)
			except Exception:
				# ignore malformed messages
				pass

	def cb_quatstamped(self, msg: QuaternionStamped):
		q = msg.quaternion
		with self.lock:
			try:
				self.latest_quat = np.array([q.w, q.x, q.y, q.z], dtype=float)
			except Exception:
				pass

	def viz_loop(self):
		"""Main visualization/update loop. Keeps viewer in sync and writes
		the latest quaternion into the cube's qpos (indices 3:7).
		"""
		while self.running and self.viewer.is_running():
			t0 = time.time()

			with self.lock:
				quat = None if self.latest_quat is None else self.latest_quat.copy()

			if quat is not None:
				# write quaternion into qpos: layout is [x,y,z, qw,qx,qy,qz]
				try:
					if self.data.qpos.shape[0] >= 7:
						self.data.qpos[3:7] = quat
						mujoco.mj_fwdPosition(self.model, self.data)
				except Exception as e:
					# log and continue
					try:
						self.get_logger().warning(f'failed to set cube quaternion: {e}')
					except Exception:
						pass

			# keep viewer responsive / synced
			try:
				self.viewer.sync()
			except Exception:
				# viewer may raise when closing; ignore
				pass

			# sleep to keep ~400Hz
			elapsed = time.time() - t0
			to_sleep = self.dt - elapsed
			if to_sleep > 0:
				time.sleep(to_sleep)

		# ensure running flag cleared
		self.running = False

	def stop(self):
		self.running = False
		try:
			if self.viz_thread.is_alive():
				self.viz_thread.join(timeout=0.5)
		except Exception:
			pass


def main(args=None):
	rclpy.init(args=args)
	node = IMUCubeVisualizer()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.get_logger().info('shutting down IMU cube visualizer')
		node.stop()
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
