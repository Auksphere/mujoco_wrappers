#!/usr/bin/env python3
"""
轻量级 IMU 测试发布器 — 以 400Hz 发布 /filter/quaternion 和 /filter/euler
用于与 `imu_test.py` 联合调试 MuJoCo 可视化。
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import QuaternionStamped, Vector3Stamped
import numpy as np
import math


class ImuPublisherTest(Node):
    def __init__(self):
        super().__init__('imu_publisher_test')
        qos = rclpy.qos.QoSProfile(depth=10)
        self.q_pub = self.create_publisher(QuaternionStamped, '/filter/quaternion', qos)
        self.e_pub = self.create_publisher(Vector3Stamped, '/filter/euler', qos)

        self.freq = 400.0
        self.dt = 1.0 / self.freq
        self.t = 0.0

        self.timer = self.create_timer(self.dt, self.timer_callback)

    def timer_callback(self):
        # generate a slowly rotating quaternion around Z
        angle = 0.5 * self.t  # rad
        qw = math.cos(angle / 2.0)
        qz = math.sin(angle / 2.0)
        q = QuaternionStamped()
        q.header.stamp = self.get_clock().now().to_msg()
        q.quaternion.w = qw
        q.quaternion.x = 0.0
        q.quaternion.y = 0.0
        q.quaternion.z = qz
        self.q_pub.publish(q)

        # publish euler (roll, pitch, yaw)
        e = Vector3Stamped()
        e.header.stamp = q.header.stamp
        e.vector.x = 0.0
        e.vector.y = 0.0
        e.vector.z = angle
        self.e_pub.publish(e)

        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = ImuPublisherTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
