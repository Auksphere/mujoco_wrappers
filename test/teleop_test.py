#!/usr/bin/env python3
"""Teleop test: load cube_ur10_scene.xml, subscribe to IMU, drive cube orientation.

Cube position is fixed in the XML; only orientation (freejoint) is updated from IMU.
UR10 is present in the scene but not controlled here.
"""

import os
import time
import threading
from typing import Optional

import numpy as np
import mujoco
import sys
import mujoco.viewer
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import QuaternionStamped
# ensure workspace src on path for local imports
script_dir = os.path.dirname(__file__)
pkg_root = os.path.abspath(os.path.join(script_dir, '..'))
src_root = os.path.abspath(os.path.join(pkg_root, '..'))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from mujoco_wrappers.controllers.ik_arm import QP
from mujoco_wrappers.controllers.util import calculate_arm_Te

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scene', 'cube_ur10_scene.xml')


class TeleopCube(Node):
    def __init__(self):
        super().__init__('teleop_cube')

        # state
        self.lock = threading.Lock()
        self.latest_quat: Optional[np.ndarray] = None  # [w,x,y,z]

        # QoS tuned for high-rate IMU
        high_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Subscribe preferred IMU topic and fallback quaternion topic
        self.create_subscription(Imu, '/imu/data', self.cb_imu, high_qos)
        self.create_subscription(QuaternionStamped, '/filter/quaternion', self.cb_quatstamped, high_qos)

        # load model
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f'model not found: {MODEL_PATH}')
            raise FileNotFoundError(MODEL_PATH)

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # --- set UR10 initial joint positions if present ---
        try:
            self.ur10_init = [0.0, -2, 2, -np.pi/2, -np.pi/2, 0.0]
            # map by joint name 'joint1'..'joint6'
            for i in range(6):
                name = f'joint{i+1}'
                # find joint index
                for jid in range(self.model.njnt):
                    try:
                        jname = self.model.joint(jid).name
                    except Exception:
                        # fallback to names array if attribute not present
                        jname = None
                    if jname == name:
                        qposadr = int(self.model.jnt_qposadr[jid])
                        if 0 <= qposadr < self.data.qpos.shape[0]:
                            self.data.qpos[qposadr] = float(self.ur10_init[i])
                        break
            # propagate to derived quantities
            mujoco.mj_fwdPosition(self.model, self.data)
        except Exception:
            self.get_logger().warning('could not set UR10 initial joint positions')

        # find cube body id
        try:
            cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b'cube')
            if cube_bid < 0:
                cube_bid = None
        except Exception:
            cube_bid = None

        # attempt to find the qpos address for the cube's freejoint
        self.cube_qpos_addr = None
        try:
            for jid in range(self.model.njnt):
                # check joint type and associated body
                jtype = int(self.model.jnt_type[jid])
                # free joint type enum
                if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                    # body id for this joint
                    bodyid = int(self.model.jnt_bodyid[jid]) if hasattr(self.model, 'jnt_bodyid') else -1
                    if cube_bid is not None and bodyid == cube_bid:
                        addr = int(self.model.jnt_qposadr[jid])
                        # freejoint occupies 7 qpos entries
                        self.cube_qpos_addr = addr
                        break
        except Exception:
            self.cube_qpos_addr = None

        # fallback: if we didn't find it, assume cube qpos are the last 7 entries
        if self.cube_qpos_addr is None:
            if self.data.qpos.shape[0] >= 7:
                self.cube_qpos_addr = int(self.data.qpos.shape[0] - 7)
                self.get_logger().warning('could not locate cube freejoint; falling back to last 7 qpos indices')
            else:
                raise RuntimeError('model qpos too small to hold cube freejoint')

        # try to start viewer, but allow headless fallback
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer_msg = 'interactive viewer available'
        except Exception:
            self.viewer = None
            viewer_msg = 'mujoco.viewer not available; running headless'

        # --- load separate UR10 model for IK (use its joints only) ---
        try:
            ur10_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ur10', 'ur10.xml')
            self.ur10_model = mujoco.MjModel.from_xml_path(ur10_path)
            self.ur10_data = mujoco.MjData(self.ur10_model)
            # IK solver instance
            self.ik_solver = QP(tol=1e-4, ilimit=200)
        except Exception:
            self.ur10_model = None
            self.ur10_data = None
            self.ik_solver = None

        # map joint qpos addresses between combined model and ur10 model for joint1..joint6
        self.robot_joint_names = [f'joint{i+1}' for i in range(6)]
        self.combined_qpos_addrs = []
        self.ur10_qpos_addrs = []
        if self.ur10_model is not None:
            for name in self.robot_joint_names:
                # combined model joint qpos addr
                try:
                    jid_c = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name.encode())
                    addr_c = int(self.model.jnt_qposadr[jid_c]) if jid_c >= 0 else None
                except Exception:
                    addr_c = None
                # ur10 model joint qpos addr
                try:
                    jid_u = mujoco.mj_name2id(self.ur10_model, mujoco.mjtObj.mjOBJ_JOINT, name.encode())
                    addr_u = int(self.ur10_model.jnt_qposadr[jid_u]) if jid_u >= 0 else None
                except Exception:
                    addr_u = None
                self.combined_qpos_addrs.append(addr_c)
                self.ur10_qpos_addrs.append(addr_u)

        # compute corresponding qvel addresses in combined model for each robot joint
        self.combined_qvel_addrs = []
        for name in self.robot_joint_names:
            try:
                jid_c = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name.encode())
                if jid_c >= 0:
                    try:
                        dofadr = int(self.model.jnt_dofadr[jid_c])
                    except Exception:
                        dofadr = None
                else:
                    dofadr = None
            except Exception:
                dofadr = None
            self.combined_qvel_addrs.append(dofadr)

        # track previous IK solution for smoothing
        self.prev_q_sol = None

        # loop control
        self.running = True
        self.dt = 0.0025  # 400Hz
        # IK solver rate: target 50Hz for expensive IK solves
        self.ik_rate = 50.0
        self.ik_dt = 1.0 / float(self.ik_rate)
        self.last_ik_ts = time.time()

        # IK targets and smoothing state
        self.q_target = None      # latest IK solution (target)
        self.q_written = None     # current smoothed written q (for interpolation)

        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

        # log initialization (viewer_msg set above)
        try:
            self.get_logger().info(f'teleop_cube initialized ({viewer_msg})')
        except Exception:
            pass

    def cb_imu(self, msg: Imu):
        q = msg.orientation
        with self.lock:
            try:
                # incoming is (x,y,z,w) in the message; store as [w,x,y,z]
                qin = np.array([q.w, q.x, q.y, q.z], dtype=float)
                # rotation of pi about Z -> quaternion r = [0,0,0,1]
                # compose r * qin (left-multiply) to apply extra rotation
                w1, x1, y1, z1 = 0.0, 0.0, 0.0, 1.0
                w2, x2, y2, z2 = qin
                w = w1*w2 - x1*x2 - y1*y2 - z1*z2
                x = w1*x2 + x1*w2 + y1*z2 - z1*y2
                y = w1*y2 - x1*z2 + y1*w2 + z1*x2
                z = w1*z2 + x1*y2 - y1*x2 + z1*w2
                qrot = np.array([w, x, y, z], dtype=float)
                # normalize to avoid drift
                norm = np.linalg.norm(qrot)
                if norm > 0:
                  qrot = qrot / norm
                self.latest_quat = qrot
            except Exception:
                pass

    def cb_quatstamped(self, msg: QuaternionStamped):
        q = msg.quaternion
        with self.lock:
            try:
                qin = np.array([q.w, q.x, q.y, q.z], dtype=float)
                # apply rotation r = [0,0,0,1] about Z (pi)
                w1, x1, y1, z1 = 0.0, 0.0, 0.0, 1.0
                w2, x2, y2, z2 = qin
                w = w1*w2 - x1*x2 - y1*y2 - z1*z2
                x = w1*x2 + x1*w2 + y1*z2 - z1*y2
                y = w1*y2 - x1*z2 + y1*w2 + z1*x2
                z = w1*z2 + x1*y2 - y1*x2 + z1*w2
                qrot = np.array([w, x, y, z], dtype=float)
                # normalize
                norm = np.linalg.norm(qrot)
                if norm > 0:
                  qrot = qrot / norm
                self.latest_quat = qrot
            except Exception:
                pass

    def loop(self):
        while self.running and (self.viewer is None or self.viewer.is_running()):
            t0 = time.time()

            t_now = t0

            with self.lock:
                quat = None if self.latest_quat is None else self.latest_quat.copy()

            if quat is not None:
                # write into qpos at cube_qpos_addr: layout [x,y,z, qw,qx,qy,qz]
                try:
                    base = self.cube_qpos_addr
                    if base is not None and base + 7 <= self.data.qpos.shape[0]:
                        self.data.qpos[base + 3: base + 7] = quat
                        mujoco.mj_fwdPosition(self.model, self.data)

                        # IK solving: only run at reduced rate (self.ik_rate). Keep IMU/cube update at full rate.
                        do_ik = (self.ik_solver is not None and self.ur10_model is not None and
                                 (t_now - self.last_ik_ts) >= self.ik_dt)

                        if do_ik:
                            try:
                                # use attachment body from combined model to get current ee position
                                ee_pos = self.data.body(self.model.body('attachment').id).xpos.copy()
                                # create target Te from ee_pos and quaternion (note: mj uses xquat)
                                # convert [w,x,y,z] to [x,y,z,w] for xquat
                                xquat = np.array([quat[0], quat[1], quat[2], quat[3]])
                                Te = calculate_arm_Te(ee_pos, xquat)

                                # prepare q_init for ur10 solver from ur10_data
                                q_init = np.zeros(self.ur10_model.nq)
                                for idx, addr in enumerate(self.ur10_qpos_addrs):
                                    if addr is not None and addr < self.ur10_data.qpos.shape[0]:
                                        q_init[addr] = float(self.ur10_data.qpos[addr])

                                q_sol, success, iters, err, jl_valid, tcost = self.ik_solver.solve(self.ur10_model, self.ur10_data, Te, self.ur10_init)
                                if success:
                                    # update IK target; keep raw solution as target
                                    self.q_target = q_sol.copy()
                                    # initialize written q if first time
                                    if self.q_written is None:
                                        self.q_written = q_sol.copy()
                                    # track prev_q_sol for optional smoothing/debug
                                    self.prev_q_sol = q_sol.copy()
                                # update last ik timestamp regardless of success to schedule next attempt
                                self.last_ik_ts = t_now
                            except Exception as e:
                                self.get_logger().warning(f'IK update failed: {e}')

                        # Between IK solves, perform low-pass interpolation toward the latest IK target
                        try:
                            if self.q_target is not None and self.q_written is not None:
                                # fraction per 400Hz step to move toward target so that roughly in ik_dt we reach it
                                beta = float(self.dt / max(self.ik_dt, self.dt))
                                # clamp
                                if beta < 0.0:
                                    beta = 0.0
                                if beta > 1.0:
                                    beta = 1.0
                                # interpolation
                                q_use = self.q_written * (1.0 - beta) + self.q_target * beta
                                self.q_written = q_use.copy()

                                # write interpolated solution back into combined model's qpos at mapped addresses
                                for i, addr_c in enumerate(self.combined_qpos_addrs):
                                    if addr_c is not None and i < len(q_use):
                                        self.data.qpos[addr_c] = float(q_use[i])
                                mujoco.mj_fwdPosition(self.model, self.data)

                                # zero joint velocities for the corresponding joints in combined model
                                for dofadr in self.combined_qvel_addrs:
                                    if dofadr is not None and 0 <= dofadr < self.data.qvel.shape[0]:
                                        self.data.qvel[dofadr] = 0.0

                                # zero global controls to prevent actuator interference
                                try:
                                    self.data.ctrl[:] = 0
                                except Exception:
                                    pass

                                # also update ur10_data qpos for next iteration
                                for i, addr_u in enumerate(self.ur10_qpos_addrs):
                                    if addr_u is not None and i < len(q_use) and addr_u < self.ur10_data.qpos.shape[0]:
                                        self.ur10_data.qpos[addr_u] = float(q_use[i])
                        except Exception:
                            pass
                except Exception as e:
                    self.get_logger().warning(f'failed writing cube quaternion: {e}')

            # sync viewer if available
            if self.viewer is not None:
                try:
                    self.viewer.sync()
                except Exception:
                    pass

            elapsed = time.time() - t0
            to_sleep = self.dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        self.running = False

    def stop(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=0.5)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = TeleopCube()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('shutting down teleop_cube')
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
