import os
import sys
import numpy as np
import mujoco
import mujoco.viewer
import time

# Ensure workspace `src` is on sys.path so local package imports work when running this
# script directly (not installed). This inserts the parent of the `mujoco_wrappers`
# directory (i.e. the workspace's `src` folder) to sys.path.
script_dir = os.path.dirname(__file__)
pkg_root = os.path.abspath(os.path.join(script_dir, '..'))
src_root = os.path.abspath(os.path.join(pkg_root, '..'))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# Use local controllers package instead of non-existent `ur10_sim` package
from mujoco_wrappers.controllers.ik_arm import QP
from mujoco_wrappers.controllers.util import calculate_arm_Te

# 路径设置
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../models/ur10/scene.xml"
)

def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 初始关节角
    print("所有关节角名称及其ID：")
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        print(f"关节{i}: 名称={joint_name}, ID={i}")
    q_cur = np.asarray((0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0))
    data.qpos[:] = q_cur
    mujoco.mj_fwdPosition(model, data)

    # 启动可视化窗口
    viewer = mujoco.viewer.launch_passive(model, data)

    # jacp = np.zeros((3, model.nv))
    # jacr = np.zeros((3, model.nv))
    # mujoco.mj_jacBodyCom(model, data, jacp, jacr, model.body("attachment").id)
    # J = np.concatenate((jacp, jacr), axis=0)
    # print("Jacobian J:\n", J)

    # 创建IK求解器（使用本仓库 controllers/ik_arm.py 中的 QP 实现）
    ik_solver = QP(tol=1e-4, ilimit=100)

    while viewer.is_running():
        # 当前末端位姿
        Te = calculate_arm_Te(data.body("attachment").xpos, data.body("attachment").xquat)
        print("当前末端位姿 Te:\n", Te[:3])
        print("请输入末端期望运动 dx dy dz (以空格分隔，回车确认，窗口ESC退出):")
        try:
            user_input = input()
        except EOFError:
            break
        if not user_input.strip():
            continue
        try:
            dx, dy, dz = map(float, user_input.strip().split())
        except Exception:
            print("输入格式错误，请重新输入。")
            continue

        # 构造目标末端位姿
        Tep = Te.copy()
        Tep[0, 3] += dx
        Tep[1, 3] += dy
        Tep[2, 3] += dz
        print("目标末端位姿 Tep:\n", Tep[:3])

        # 求解逆运动学
        q_sol, success, iterations, error, jl_valid, total_t = ik_solver.solve(model, data, Tep, q_cur)
        print("IK求解成功:", success)
        print("残差误差:", error)
        print("关节限位有效:", jl_valid)
        print("总耗时(s):", total_t)

        if not success:
            print("逆运动学求解失败，请重新输入。")
            continue

        # 插值更新关节角
        steps = 1000
        q_traj = np.linspace(q_cur, q_sol, steps)
        for q in q_traj:
            data.qpos[:] = q
            mujoco.mj_fwdPosition(model, data)
            viewer.sync()
            mujoco.mj_step(model, data)
        q_cur = q_sol.copy()

        # 检查正向运动学是否到达目标
        data.qpos[:] = q_cur
        mujoco.mj_fwdPosition(model, data)
        Te_check = calculate_arm_Te(data.body("attachment").xpos, data.body("attachment").xquat)
        print("IK解对应的末端位姿:\n", Te_check)
        print("末端位置误差:", np.linalg.norm(Te_check[:3, 3] - Tep[:3, 3]))

        # 等待下一次输入

    print("可视化窗口已关闭，程序结束。")

if __name__ == "__main__":
    main()