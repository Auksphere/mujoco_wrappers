import numpy as np


# Source of truth for VAC hyperparameters.
VAC_HYPERPARAMS = {
    "duration_sec": 15.0,
    "success_xy_threshold": 0.008,
    "success_z_threshold": 0.012,
    "distance_threshold": 0.05,
    "hole_position": np.array([0.0, -0.7, 0.122], dtype=np.float64),
    "d_far": 0.1,
    "d_near": 5.0,
    "d_default": 0.1,
    "Mc_diag": np.array([1500.0, 1500.0, 1500.0, 500.0, 500.0, 500.0], dtype=np.float64),
    "Kc_far_diag": np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float64),
    "Kc_near_diag": np.array([200.0, 200.0, 3500.0, 200.0, 200.0, 200.0], dtype=np.float64),
    "wrench_bias_steps": 500,
    "use_wrench_bias_prior": True,
    "wrench_bias_prior": np.array(
        [0.239438, 0.039126, -10.410907, 0.003842, 0.026428, 0.001005],
        dtype=np.float64,
    ),
}


def get_vac_hyperparams():
    cfg = dict(VAC_HYPERPARAMS)
    cfg["hole_position"] = cfg["hole_position"].copy()
    cfg["wrench_bias_prior"] = cfg["wrench_bias_prior"].copy()
    cfg["Mc"] = np.diag(cfg["Mc_diag"])
    cfg["Kc_far"] = np.diag(cfg["Kc_far_diag"])
    cfg["Kc_near"] = np.diag(cfg["Kc_near_diag"])
    return cfg
