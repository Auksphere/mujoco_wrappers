import numpy as np
import sympy as sp
import mujoco


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

def calculate_desired_pose_trajectory(task, duration = 10.0):
    """
    Defines the symbolic trajectory for the end-effector based on task type.
    
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
    
    Trajectory Details:
        pih (peg-in-hole): Two-phase smooth trajectory
            - Phase 1 (0 to 60% duration): Move from start position to hole entrance
            - Phase 2 (60% to 100% duration): Insert from entrance to target inside hole
            - Uses quintic polynomial interpolation for smooth velocity profiles
            
        regulation: Static regulation at fixed position
    """
    t = sp.symbols('t')
    max_time = duration
    
    # Define default position and orientation based on task
    if task == "regulation":
        pd_default = np.array([0.0, -0.7, 0.3])
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    elif task == "pih":
        pd_default = np.array([0.0, -0.70, 0.25])  
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        
    else:
        raise ValueError(f"Invalid task: {task}")
    
    pd_default_sym = sp.Matrix([float(x) for x in pd_default])
    Rd_default_sym = sp.Matrix([[float(x) for x in row] for row in Rd_default])
    
    # Define trajectory based on task type
    if task == 'regulation':
        pd_t_sim = pd_default_sym
        Rd_t_sim = Rd_default_sym
        
    elif task == 'pih':
        # Define key waypoints for two-phase trajectory
        start_position = pd_default_sym  
        hole_entrance = sp.Matrix([0, -0.702, 0.17])  
        hole_target = sp.Matrix([0, -0.7, 0.11]) 
        
        t_transition = 0.5 * max_time  # Transition time between phases

        phase_weight = sp.tanh(10 * (t - t_transition))  # Smooth transition around t_transition
        phase_weight = (phase_weight + 1) / 2  # Map from [-1,1] to [0,1]
        
        # Phase 1: start -> entrance trajectory
        t1_norm = sp.Min(t / t_transition, 1)
        s1 = 6*t1_norm**5 - 15*t1_norm**4 + 10*t1_norm**3
        phase1_pos = start_position + s1 * (hole_entrance - start_position)
        
        # Phase 2: entrance -> target trajectory  
        t2_norm = sp.Max(0, (t - t_transition) / (max_time - t_transition))
        s2 = 6*t2_norm**5 - 15*t2_norm**4 + 10*t2_norm**3
        phase2_pos = hole_entrance + s2 * (hole_target - hole_entrance)
        
        # Blend between the two phases
        pd_t_sim = (1 - phase_weight) * phase1_pos + phase_weight * phase2_pos
        
        # Orientation remains constant (pointing down)
        Rd_t_sim = Rd_default_sym


    pd_t = sp.lambdify(t, pd_t_sim, "numpy")
    Rd_t = sp.lambdify(t, Rd_t_sim, "numpy")

    # Differentiate with symbolic expressions
    dpd_t_sim = sp.diff(pd_t_sim, t)
    dRd_t_sim = sp.diff(Rd_t_sim, t)
    ddpd_t_sim = sp.diff(dpd_t_sim, t)
    ddRd_t_sim = sp.diff(dRd_t_sim, t)

    # Convert symbolic to numpy expressions
    pd_t = sp.lambdify(t, pd_t_sim, "numpy")
    Rd_t = sp.lambdify(t, Rd_t_sim, "numpy")
    dpd_t = sp.lambdify(t, dpd_t_sim, "numpy")
    dRd_t = sp.lambdify(t, dRd_t_sim, "numpy")
    ddpd_t = sp.lambdify(t, ddpd_t_sim, "numpy")
    ddRd_t = sp.lambdify(t, ddRd_t_sim, "numpy")

    return pd_t, Rd_t, dpd_t, dRd_t, ddpd_t, ddRd_t


def get_initial_joint_config(task, xml_file, ik_class, mode=None):
    """Compute an initial joint configuration using IK.

    Parameters
    ----------
    task : str
        Task name, e.g. "pih" or "regulation".
    xml_file : str
        Path to the MuJoCo XML model file.
    ik_class : type
        IK solver class (e.g. controllers.ik_arm.IKArm) providing a ``solve`` method.
    mode : str or None
        If "test", q is deterministic. Otherwise, use the current
        global RNG state (no seeding here).
    """

    try:
        # Use deterministic random seed in test mode
        if mode == "test":
            # np.random.seed(42)
            q_test = np.array([-1.7649302348, 1.4444537573, -1.9586744434, 
                               2.0937486000, 1.5705321056, 1.3857693592])
            return q_test

        # Get the trajectory functions which contain the default pose information
        pd_t, Rd_t, _, _, _, _ = calculate_desired_pose_trajectory(task, 0.1)

        # Default orientation
        Rd_default = np.array(Rd_t(0.0)).reshape(3, 3)

        # Random starting position in a cylindrical region around center
        center = np.array([0.0, -0.70, 0.25])

        theta = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, 0.4)
        z_offset = np.random.uniform(-0.05, 0.25)

        pd_random = center + np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            z_offset,
        ])

        print(f"[INFO] Random starting position: {pd_random}, task='{task}'")
        print(
            f"[INFO] Offset from center {center}: "
            f"[{pd_random[0]-center[0]:.3f}, {pd_random[1]-center[1]:.3f}, {pd_random[2]-center[2]:.3f}]"
        )

        # Target transform
        T_target = np.eye(4)
        T_target[:3, :3] = Rd_default
        T_target[:3, 3] = pd_random

        # Temporary model and IK solver
        temp_model = mujoco.MjModel.from_xml_path(xml_file)
        temp_data = mujoco.MjData(temp_model)

        temp_ik_solver = ik_class(solver_type="QP", tol=1e-4, ilimit=5000)

        q_guess = np.array(
            [
                -1.75916382,
                1.27408681,
                -2.06908278,
                2.35226387,
                1.57079097,
                1.38243476,
            ]
        )

        q_sol, success, *_ = temp_ik_solver.solve(temp_model, temp_data, T_target, q_guess)

        if success:
            return q_sol

    except Exception as e:
        print(f"[WARN] get_initial_joint_config failed, using fallback: {e}")

    # Fallback configurations
    fallback_configs = {
        "pih": np.array(
            [
                -1.75916382,
                1.27408681,
                -2.06908278,
                2.35226387,
                1.57079097,
                1.38243476,
            ]
        ),
        "regulation": np.array(
            [
                -1.7671236707,
                1.5031112517,
                -1.8753320848,
                1.9429508990,
                1.5703766827,
                1.3875707847,
            ]
        ),
    }

    return fallback_configs.get(
        task,
        np.array([0.0, np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0]),
    )

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