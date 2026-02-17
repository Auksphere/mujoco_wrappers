import numpy as np
import mujoco
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath('.'))

# Load IK solver
import importlib.util
spec_util = importlib.util.spec_from_file_location('util', 'controllers/util.py')
util_module = importlib.util.module_from_spec(spec_util)
sys.modules['controllers.util'] = util_module
spec_util.loader.exec_module(util_module)

spec = importlib.util.spec_from_file_location('controllers.ik_arm', 'controllers/ik_arm.py')
ik_arm_module = importlib.util.module_from_spec(spec)
sys.modules['controllers.ik_arm'] = ik_arm_module
spec.loader.exec_module(ik_arm_module)
IKArm = ik_arm_module.IKArm

model = mujoco.MjModel.from_xml_path('models/jaka_zu12/jaka_admittance.xml')
data = mujoco.MjData(model)
ik_solver = IKArm(solver_type='QP', tol=1e-4, ilimit=500)

def get_trajectory_start(task):
    if task == 'regulation':
        return np.array([0.0, -0.7, 0.22]), np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    elif task == 'circle':
        return np.array([0.1, -0.7, 0.3]), np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    elif task == 'line':
        return np.array([-0.25, -0.7, 0.3]), np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    elif task == 'sphere':
        pd_default = np.array([0.0, -0.7, -0.028])
        theta = -np.pi / 6
        r_sphere = 0.110
        pd = pd_default + np.array([r_sphere * np.sin(theta), 0, r_sphere * np.cos(theta)])
        c, s = np.cos(theta), np.sin(theta)
        Rd = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return pd, Rd

def check_manip(q):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, model.body('attachment').id)
    jac = np.vstack([jacp[:, :6], jacr[:, :6]])
    return np.sqrt(np.linalg.det(jac @ jac.T)), np.linalg.cond(jac)

guesses = [
    # np.array([0.0, -1.0, 1.0, 0.5, 0.5, 0.5]),
    # np.array([0.3, -1.2, 0.8, 0.3, 0.3, 0.3]),
    # np.array([-0.3, -1.2, 0.8, 0.3, 0.3, 0.3]),
    # np.array([0.0, -0.9, 1.2, 0.8, 0.8, 0.8]),
    # np.array([0.5, -1.1, 1.0, 0.4, 0.6, 0.4]),
    # np.array([-0.5, -1.1, 1.0, 0.4, 0.6, 0.4]),
    # np.array([-2.0, 1.2, -2.0, 2.15, 2.1, 1.1]),
    np.array([-2.0, 1.5, -1.8, 1.85, 1.5, 1.1]),
    np.array([-1.7, 1.6, -1.9, 1.8, 1.57, 1.4]),
]

print('='*80)
print('Finding configurations with POSITIVE joints 4, 5, 6')
print('='*80)

results = {}
for task in ['regulation', 'circle', 'line', 'sphere']:
    print(f'\n{task.upper()}:')
    pd_start, Rd_start = get_trajectory_start(task)
    Tep = np.eye(4)
    Tep[:3, :3] = Rd_start
    Tep[:3, 3] = pd_start
    
    candidates = []
    for i, guess in enumerate(guesses):
        q_sol, success, iters, error, jl_valid, solve_time = ik_solver.solve(model, data, Tep, guess)
        if success and q_sol[3] > 0 and q_sol[4] > 0 and q_sol[5] > 0:
            data.qpos[:6] = q_sol
            mujoco.mj_forward(model, data)
            body_pos = data.xpos[model.body('attachment').id]
            pos_error = np.linalg.norm(body_pos - pd_start)
            if pos_error < 0.01:
                manip, cond = check_manip(q_sol)
                candidates.append({'q': q_sol, 'pos_error': pos_error, 'manip': manip, 'cond': cond})
                print(f'  ✓ j4,j5,j6=[{q_sol[3]:.2f},{q_sol[4]:.2f},{q_sol[5]:.2f}] PosErr={pos_error*1000:.1f}mm')
    
    if candidates:
        candidates.sort(key=lambda x: (-x['manip'], x['pos_error']))
        best = candidates[0]
        results[task] = best['q']
        print(f'  BEST: Manip={best["manip"]:.4f}, PosErr={best["pos_error"]*1000:.1f}mm')

if results:
    print('\n' + '='*80)
    print('SUCCESS! Copy these configurations:')
    print('='*80)
    for task, sol in results.items():
        sol_str = ', '.join([f'{x:.10f}' for x in sol])
        print(f'"{task}": np.array([{sol_str}]),')