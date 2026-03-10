import numpy as np
import sympy as sp
import mujoco
from scipy.linalg import expm
from scipy.spatial.transform import Rotation, Slerp


def vee_map(R):
    v3 = -R[0, 1]
    v1 = -R[1, 2]
    v2 = R[0, 2]
    return np.array([v1, v2, v3]).reshape((-1, 1))


def hat_map(w):
    w = w.reshape((-1,))
    w_hat = np.array(
        [
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0],
        ]
    )
    return w_hat

def calculate_desired_pose_trajectory(task, duration = 15.0, Rd_default_override=None, initial_config = None):
    """
    Defines a nominal symbolic pose reference for the end-effector.
    
    Args:
        task (str): Task type - 'regulation' or 'pih' (peg-in-hole)
        duration (float): Total trajectory duration in seconds
        
    Returns:
        tuple: (pd_t, Rd_t, dpd_t, dRd_t, ddpd_t, ddRd_t)
            - pd_t: Position trajectory function pd(t)
            - Rd_t: Orientation trajectory function Rd(t) 
            - dpd_t: Velocity trajectory function dpd/dt(t)
            - dRd_t: Angular velocity trajectory function dRd/dt(t)
            - ddpd_t: Acceleration trajectory function d²pd/dt²(t)
            - ddRd_t: Angular acceleration trajectory function d²Rd/dt²(t)
    """
    t = sp.symbols('t')
    max_time = duration

    # Define default position and orientation based on task
    if task == "regulation":
        pd_default = np.array([0.0, -0.70, 0.3])
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    elif task == "pih":
        pd_default = np.array([0.0, -0.70, 0.25])
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    else:
        raise ValueError(f"Invalid task: {task}")

    if Rd_default_override is not None:
        Rd_default = np.array(Rd_default_override).reshape(3, 3)

    pd_default_sym = sp.Matrix([float(x) for x in pd_default])
    Rd_default_sym = sp.Matrix([[float(x) for x in row] for row in Rd_default])
    Rd_default_np = np.array(Rd_default, dtype=float).reshape(3, 3)

    # Define trajectory based on task type
    if task == 'regulation':
        pd_t_sim = pd_default_sym
        Rd_t_sim = Rd_default_sym
        start_orientation_np = Rd_default_np
        hole_orientation_np = Rd_default_np
        t_transition = 0.5 * max_time

    elif task == 'pih':
        # initial_config:
        #   - None
        #   - dict: {"position": (3,), "orientation": (3,3)}
        start_position = pd_default_sym
        start_orientation = Rd_default_sym
        start_orientation_np = Rd_default_np

        if initial_config is not None:
            if isinstance(initial_config, dict):
                if "position" in initial_config:
                    p = np.asarray(initial_config["position"], dtype=float).reshape(-1)
                    if p.size != 3:
                        raise ValueError(f"initial_config['position'] must have 3 elements, got {p.shape}")
                    start_position = sp.Matrix(p.tolist())

                if "orientation" in initial_config:
                    R0 = np.asarray(initial_config["orientation"], dtype=float).reshape(3, 3)
                    start_orientation = sp.Matrix([[float(x) for x in row] for row in R0])
                    start_orientation_np = R0

            else:
                arr = np.asarray(initial_config, dtype=float).reshape(-1)
                if arr.size == 3:
                    start_position = sp.Matrix(arr.tolist())
                elif arr.size == 12:
                    start_position = sp.Matrix(arr[:3].tolist())
                    R0 = arr[3:].reshape(3, 3)
                    start_orientation_np = R0
                else:
                    raise ValueError(
                        f"initial_config shape err"
                    )

        hole_entrance = sp.Matrix([-0.00, -0.7022, 0.172])
        hole_target = sp.Matrix([0.0, -0.7, 0.122])
        hole_orientation = Rd_default_sym
        hole_orientation_np = Rd_default_np

        t_transition = 0.5 * max_time  # Transition time between phases

        phase_weight = sp.tanh(10 * (t - t_transition))
        phase_weight = (phase_weight + 1) / 2

        t1_norm = sp.Min(t / t_transition, 1)
        s1 = 6*t1_norm**5 - 15*t1_norm**4 + 10*t1_norm**3
        phase1_pos = start_position + s1 * (hole_entrance - start_position)

        t2_norm = sp.Max(0, (t - t_transition) / (max_time - t_transition))
        s2 = 6*t2_norm**5 - 15*t2_norm**4 + 10*t2_norm**3
        phase2_pos = hole_entrance + s2 * (hole_target - hole_entrance)

        pd_t_sim = (1 - phase_weight) * phase1_pos + phase_weight * phase2_pos

        # Keep symbolic placeholder for compatibility; runtime Rd_t uses SO(3) Slerp below.
        Rd_t_sim = hole_orientation

    pd_t = sp.lambdify(t, pd_t_sim, "numpy")

    dpd_t_sim = sp.diff(pd_t_sim, t)
    dRd_t_sim = sp.diff(Rd_t_sim, t)
    ddpd_t_sim = sp.diff(dpd_t_sim, t)
    ddRd_t_sim = sp.diff(dRd_t_sim, t)

    pd_t = sp.lambdify(t, pd_t_sim, "numpy")
    dpd_t = sp.lambdify(t, dpd_t_sim, "numpy")
    ddpd_t = sp.lambdify(t, ddpd_t_sim, "numpy")

    key_times = np.array([0.0, float(t_transition)], dtype=float)
    key_rots = Rotation.from_matrix(
        np.stack([start_orientation_np, hole_orientation_np], axis=0)
    )
    slerp = Slerp(key_times, key_rots)

    def _clip_time(t_eval):
        return float(np.clip(float(t_eval), 0.0, max_time))

    def _phase1_u(t_eval):
        tt = _clip_time(t_eval)
        if t_transition <= 1e-9:
            return 1.0
        tau = min(tt / t_transition, 1.0)
        return 6.0 * tau**5 - 15.0 * tau**4 + 10.0 * tau**3

    def Rd_t(t_eval):
        u = _phase1_u(t_eval)
        return slerp([u * t_transition]).as_matrix()[0]

    def dRd_t(t_eval, h=1e-3):
        t0 = _clip_time(t_eval)
        tp = min(t0 + h, max_time)
        tm = max(t0 - h, 0.0)
        denom = max(tp - tm, 1e-9)
        return (Rd_t(tp) - Rd_t(tm)) / denom

    def ddRd_t(t_eval, h=1e-3):
        t0 = _clip_time(t_eval)
        tp = min(t0 + h, max_time)
        tm = max(t0 - h, 0.0)
        dt1 = max(tp - t0, 1e-9)
        dt2 = max(t0 - tm, 1e-9)
        return (Rd_t(tp) - 2.0 * Rd_t(t0) + Rd_t(tm)) / max((0.5 * (dt1 + dt2))**2, 1e-9)

    return pd_t, Rd_t, dpd_t, dRd_t, ddpd_t, ddRd_t


def get_initial_joint_config(task, xml_file, ik_class, mode=None):
    """
    Compute an initial joint config using IK.
    return fixed joint config if mode == 'test'
    """

    fallback_q = np.array([
        -1.75916382,
        1.27408681,
        -2.06908278,
        2.35226387,
        1.57079097,
        1.38243476,
    ])
    fallback_pose = {
    "position": np.array([0.0, -0.7, 0.25], dtype=float),
    "orientation": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float),
    }
    try:
        # Use fixed pose in test mode
        if mode == "test":
            q_test = np.array([-1.7649302348, 1.4444537573, -1.9586744434, 
                               2.0937486000, 1.5705321056, 1.3857693592])
            return q_test, {"position": np.array([0.0, -0.7, 0.25]),"orientation":np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])}

        if task == "regulation":
            Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
        elif task == "pih":
            Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
        else:
            raise ValueError(f"Invalid task: {task}")
        
        center = np.array([0.0, -0.70, 0.25])

        theta = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, 0.4)
        z_offset = np.random.uniform(-0.05, 0.25)

        pd_random = center + np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            z_offset,
        ])

        max_angle = np.deg2rad(45.0)
        axis = np.random.normal(size=3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / axis_norm
        angle = np.random.uniform(-max_angle, max_angle)
        w = axis * angle
        R_delta = expm(hat_map(w))
        Rd_random = R_delta @ Rd_default

        # Target transform
        T_target = np.eye(4)
        T_target[:3, :3] = Rd_random
        T_target[:3, 3] = pd_random

        temp_model = mujoco.MjModel.from_xml_path(xml_file)
        temp_data = mujoco.MjData(temp_model)

        temp_ik_solver = ik_class(solver_type="QP", tol=1e-4, ilimit=5000)

        q_guess = fallback_q.copy()

        q_sol, success, *_ = temp_ik_solver.solve(temp_model, temp_data, T_target, q_guess)

        if success:
            return q_sol, {"position": pd_random, "orientation": Rd_random}

        return fallback_q, fallback_pose

    except Exception as e:
        print(f"[WARN] get_initial_joint_config failed, using fallback: {e}")
        return fallback_q, fallback_pose


def rotmat_x(th):
    R = np.array([[1,0,0],
                    [0,np.cos(th),-np.sin(th)],
                    [0,np.sin(th), np.cos(th)]])

    return R

def adjoint_g_ed(g_ed):
    p = g_ed[:3,3]
    R = g_ed[:3,:3]

    p_hat = hat_map(p)
    # translation part first adjoint map
    adj = np.zeros((6,6))
    adj[:3,:3] = R
    adj[3:,3:] = R
    adj[:3,3:] = p_hat @ R

    return adj

def adjoint_g(g):
    R = g[:3, :3]
    p = g[:3, 3]
    Ad_g = np.zeros((6, 6))
    Ad_g[:3, :3] = R
    Ad_g[3:, 3:] = R
    Ad_g[:3, 3:] = hat_map(p) @ R
    return Ad_g

def adjoint_g_ed_dual(g_ed):
    mat = adjoint_g_ed(np.linalg.inv(g_ed))

    return mat.T

def adjoint_g_ed_deriv(g, gd, v, w, vd, wd):
    v = v.reshape((-1,1))
    w = w.reshape((-1,1))
    vd = vd.reshape((-1,1))
    wd = wd.reshape((-1,1))

    g_ed = np.linalg.inv(g) @ gd
    p_ed = g_ed[:3,3]
    R_ed = g_ed[:3,:3]

    mat = np.zeros((6,6))

    dR_ed = hat_map(w) @ R_ed - R_ed @ hat_map(wd)
    dp_ed = -v - hat_map(w) @ p_ed + R_ed @ vd

    mat[:3, :3] = dR_ed
    mat[:3, 3:] = hat_map(p_ed)@ dR_ed + hat_map(dp_ed) @ R_ed
    mat[3:, 3:] = dR_ed

    return mat