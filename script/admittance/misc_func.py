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
    """
    t = sp.symbols('t')
    max_time = duration
    
    # Define default position and orientation based on task
    if task == "regulation":
        pd_default = np.array([0.0, -0.7, 0.3])
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        
    elif task == "circle":
        pd_default = np.array([0.0, -0.7, 0.3])
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        
    elif task == "line":
        pd_default = np.array([0.0, -0.7, 0.3])
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        
    elif task == "sphere":
        pd_default = np.array([0.0, -0.7, -0.028])  # Center of the sphere
        Rd_default = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    else:
        raise ValueError(f"Invalid task: {task}")
    
    pd_default_sym = sp.Matrix([float(x) for x in pd_default])
    Rd_default_sym = sp.Matrix([[float(x) for x in row] for row in Rd_default])
    
    # Define trajectory based on task type
    if task == 'regulation':
        pd_t_sim = pd_default_sym
        Rd_t_sim = Rd_default_sym
        
    elif task == 'circle':
        r = 0.1
        pd_t_sim = pd_default_sym + sp.Matrix([r * sp.cos(t), r * sp.sin(t), 0])
        Rd_t_sim = Rd_default_sym
        
    elif task == 'line':
        pd_t_sim = pd_default_sym + sp.Matrix([0.05 * (t - max_time / 2), 0, 0])
        Rd_t_sim = Rd_default_sym
        
    elif task == 'sphere':
        total_radian = np.pi / 3
        omega_value = total_radian / max_time
        theta = omega_value * t - total_radian * 0.5
        r_sphere = 0.1002
        pd_t_sim = pd_default_sym + sp.Matrix([r_sphere * sp.sin(theta), 0, r_sphere * sp.cos(theta)])
        rotmat_y = sp.Matrix([[1, 0, 0], [0, sp.cos(theta), -sp.sin(theta)], [0, sp.sin(theta), sp.cos(theta)]])
        Rd_t_sim = Rd_default_sym @ rotmat_y

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