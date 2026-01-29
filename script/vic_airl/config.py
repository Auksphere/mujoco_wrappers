"""
Configuration file for AIRL Variable Impedance Control Learning

Modify these parameters to tune the training process.
"""

# =============================================================================
# Environment Configuration
# =============================================================================

ENV_CONFIG = {
    'xml_path': 'models/jaka_zu12/jaka_pih.xml',
    'control_dt': 0.04,          # 25Hz control frequency
    'physics_dt': 0.001,         # 1kHz simulation frequency
    'max_episode_steps': 500,    # Maximum steps per episode
    'success_threshold': 0.01,   # 1cm for successful insertion
    'max_force': 50.0,           # Maximum allowed force (N)
}

# =============================================================================
# Expert Demonstration Configuration
# =============================================================================

EXPERT_CONFIG = {
    'n_demonstrations': 10,      # Number of expert demonstrations to generate
    'duration': 10.0,            # Duration of each demonstration (seconds)
    'save_path': 'data/expert_demonstrations.pkl',
    
    # Expert impedance parameters (hand-tuned)
    'expert_M': [20.0, 20.0, 20.0, 5.0, 5.0, 5.0],         # Mass
    'expert_D': [15.0, 15.0, 30.0, 8.0, 8.0, 8.0],         # Damping (higher in Z)
    'expert_K': [800.0, 800.0, 400.0, 200.0, 200.0, 200.0], # Stiffness (lower in Z)
    
    # Initial configuration
    'initial_q': [0.0, 0.7854, -0.7854, 0.0, 0.7854, 0.0],  # ~[0, π/4, -π/4, 0, π/4, 0]
}

# =============================================================================
# AIRL Training Configuration
# =============================================================================

AIRL_CONFIG = {
    # Network architecture
    'state_dim': 34,
    'action_dim': 12,
    'hidden_dim': 256,
    
    # Training hyperparameters
    'n_epochs': 1000,
    'n_episodes_per_epoch': 10,
    'batch_size': 512,           # Increased for better GPU utilization
    'buffer_size': 100000,
    
    # Learning rates - better balanced for AIRL
    'lr_policy': 1e-4,           # Slightly higher for policy
    'lr_discriminator': 5e-5,    # Lower for discriminator
    
    # RL parameters
    'gamma': 0.99,              # Discount factor
    'tau': 0.005,               # Soft update parameter
    
    # Update frequencies - more balanced for AIRL
    'discriminator_updates': 10,  # Reduced to prevent overfitting
    'policy_updates': 50,         # Increased to help policy learn better
    
    # Regularization
    'entropy_coef': 0.01,       # Entropy bonus coefficient
    
    # Model saving
    'save_dir': 'script/models/airl',
    'save_freq': 100,           # Save checkpoint every N epochs
    'eval_freq': 10,            # Evaluate every N epochs
}

# =============================================================================
# Impedance Parameter Ranges
# =============================================================================

IMPEDANCE_RANGES = {
    # Linear stiffness (N/m)
    'kp_linear_min': 100.0,
    'kp_linear_max': 2000.0,
    
    # Linear damping (Ns/m)
    'kd_linear_min': 10.0,
    'kd_linear_max': 200.0,
    
    # Angular stiffness (Nm/rad)
    'kp_angular_min': 50.0,
    'kp_angular_max': 500.0,
    
    # Angular damping (Nms/rad)
    'kd_angular_min': 5.0,
    'kd_angular_max': 100.0,
}

# =============================================================================
# Evaluation Configuration
# =============================================================================

EVAL_CONFIG = {
    'n_eval_episodes': 10,
    'render': False,
    'save_trajectory': True,
    'plot_dir': 'plots',
}

# =============================================================================
# IK Solver Configuration
# =============================================================================

IK_CONFIG = {
    'ee_name': 'end_effector',
    'robot_name': 'jaka',
    'ilimit': 50,               # Max iterations
    'tol': 1e-4,                # Tolerance
    'reject_jl': True,          # Reject joint limit violations
    'ps': 0.1,                  # Joint limit safety margin
}

# =============================================================================
# Trajectory Configuration
# =============================================================================

TRAJECTORY_CONFIG = {
    'task': 'peg_insertion',
    
    # Start position (above hole)
    'start_pos': [0.5, 0.0, 0.35],
    
    # Goal position (inside hole)
    'goal_pos': [0.5, 0.0, 0.25],
    
    # Orientation (pointing down: roll=0, pitch=π, yaw=0)
    'orientation': [0.0, 3.14159, 0.0],  # [roll, pitch, yaw]
}

# =============================================================================
# Logging and Visualization
# =============================================================================

LOGGING_CONFIG = {
    'verbose': True,
    'log_freq': 10,             # Log every N epochs
    'tensorboard': False,       # Use TensorBoard logging
    'wandb': False,             # Use Weights & Biases logging
}

# =============================================================================
# Hardware/Performance Configuration
# =============================================================================

HARDWARE_CONFIG = {
    'device': 'auto',           # 'cuda', 'cpu', or 'auto'
    'num_workers': 4,           # For parallel data collection
    'pin_memory': True,         # Pin memory for faster GPU transfer
}


def get_config():
    """
    Get complete configuration dictionary.
    
    Returns:
        dict: Complete configuration
    """
    return {
        'env': ENV_CONFIG,
        'expert': EXPERT_CONFIG,
        'airl': AIRL_CONFIG,
        'impedance': IMPEDANCE_RANGES,
        'eval': EVAL_CONFIG,
        'ik': IK_CONFIG,
        'trajectory': TRAJECTORY_CONFIG,
        'logging': LOGGING_CONFIG,
        'hardware': HARDWARE_CONFIG,
    }


def print_config():
    """Print current configuration."""
    config = get_config()
    
    print("="*70)
    print("AIRL Variable Impedance Control - Configuration")
    print("="*70)
    
    for section, params in config.items():
        print(f"\n{section.upper()}")
        print("-" * 70)
        for key, value in params.items():
            print(f"  {key:30s}: {value}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print_config()
