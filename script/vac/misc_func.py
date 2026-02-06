import numpy as np
import sympy as sp

def vee_map(R):
    v3 = -R[0,1]
    v1 = -R[1,2]
    v2 = R[0,2]
    return np.array([v1,v2,v3]).reshape((-1,1))

def hat_map(w):
    w = w.reshape((-1,))
    w_hat = np.array([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])
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
        pd_default = np.array([0.0, -0.70, 0.15])  
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
        hole_entrance = sp.Matrix([0, -0.702, 0.08])  
        hole_target = sp.Matrix([0, -0.7, 0.01]) 
        
        t_transition = 0.7 * max_time  # Transition time between phases
        
        # s_total = 6*(t/max_time)**5 - 15*(t/max_time)**4 + 10*(t/max_time)**3
        
        # Weight for phase selection (0 for phase 1, 1 for phase 2)
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


def vee_map(R):
    v3 = -R[0,1]
    v1 = -R[1,2]
    v2 = R[0,2]
    return np.array([v1,v2,v3]).reshape((-1,1))

def hat_map(w):
    w = w.reshape((-1,))
    w_hat = np.array([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])
    return w_hat

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