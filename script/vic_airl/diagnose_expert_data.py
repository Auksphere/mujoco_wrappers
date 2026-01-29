#!/usr/bin/env python3
"""
Diagnose expert demonstration data quality
"""

import pickle
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from envs.peg_in_hole_env import PegInHoleEnv

def diagnose_expert_data(expert_path='data/expert_demonstrations.pkl'):
    """Diagnose expert data quality."""
    
    # Load expert data
    print(f"Loading expert data from {expert_path}...")
    with open(expert_path, 'rb') as f:
        expert_demos = pickle.load(f)
    
    print(f"\n{'='*70}")
    print(f"EXPERT DATA DIAGNOSIS")
    print(f"{'='*70}\n")
    
    # Basic statistics
    print(f"Number of demonstrations: {len(expert_demos)}")
    
    trajectory_lengths = [len(demo) for demo in expert_demos]
    print(f"\nTrajectory lengths:")
    print(f"  Min: {min(trajectory_lengths)}")
    print(f"  Max: {max(trajectory_lengths)}")
    print(f"  Mean: {np.mean(trajectory_lengths):.1f}")
    print(f"  Std: {np.std(trajectory_lengths):.1f}")
    
    # Check data structure
    print(f"\n{'='*70}")
    print(f"DATA STRUCTURE CHECK")
    print(f"{'='*70}\n")
    
    first_transition = expert_demos[0][0]
    print(f"Keys in transition: {first_transition.keys()}")
    print(f"\nData shapes:")
    print(f"  state: {first_transition['state'].shape}")
    print(f"  action: {first_transition['action'].shape}")
    print(f"  next_state: {first_transition['next_state'].shape if first_transition['next_state'] is not None else 'None'}")
    
    # Check state distribution
    print(f"\n{'='*70}")
    print(f"STATE DISTRIBUTION")
    print(f"{'='*70}\n")
    
    all_states = []
    all_actions = []
    all_forces = []
    all_ee_pos = []
    
    for demo in expert_demos:
        for trans in demo:
            all_states.append(trans['state'])
            all_actions.append(trans['action'])
            if 'force' in trans:
                all_forces.append(trans['force'])
            if 'ee_pos' in trans:
                all_ee_pos.append(trans['ee_pos'])
    
    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    
    print(f"State statistics (first 10 dims):")
    for i in range(min(10, all_states.shape[1])):
        print(f"  Dim {i:2d}: mean={all_states[:, i].mean():8.4f}, "
              f"std={all_states[:, i].std():8.4f}, "
              f"min={all_states[:, i].min():8.4f}, "
              f"max={all_states[:, i].max():8.4f}")
    
    # Check action distribution
    print(f"\n{'='*70}")
    print(f"ACTION DISTRIBUTION (Normalized impedance params)")
    print(f"{'='*70}\n")
    
    action_names = [
        'kp_x', 'kp_y', 'kp_z',
        'kd_x', 'kd_y', 'kd_z',
        'kp_roll', 'kp_pitch', 'kp_yaw',
        'kd_roll', 'kd_pitch', 'kd_yaw'
    ]
    
    for i, name in enumerate(action_names):
        print(f"  {name:10s}: mean={all_actions[:, i].mean():6.4f}, "
              f"std={all_actions[:, i].std():6.4f}, "
              f"min={all_actions[:, i].min():6.4f}, "
              f"max={all_actions[:, i].max():6.4f}")
    
    # Check if actions are constant (expert uses fixed impedance)
    action_std = all_actions.std(axis=0)
    if np.all(action_std < 1e-6):
        print(f"\n⚠️  WARNING: All actions are nearly constant!")
        print(f"   This means expert uses fixed impedance parameters.")
        print(f"   AIRL may struggle to learn variable impedance policy.")
    
    # Check forces
    if len(all_forces) > 0:
        all_forces = np.array(all_forces)
        max_forces = np.linalg.norm(all_forces, axis=1)
        
        print(f"\n{'='*70}")
        print(f"FORCE ANALYSIS")
        print(f"{'='*70}\n")
        print(f"Max force magnitude:")
        print(f"  Mean: {max_forces.mean():.2f}N")
        print(f"  Std: {max_forces.std():.2f}N")
        print(f"  Max: {max_forces.max():.2f}N")
        print(f"  Min: {max_forces.min():.2f}N")
        
        if max_forces.max() > 100:
            print(f"\n⚠️  WARNING: Very high contact forces detected!")
            print(f"   Max force: {max_forces.max():.2f}N")
    
    # Check end-effector positions
    if len(all_ee_pos) > 0:
        all_ee_pos = np.array(all_ee_pos)
        
        print(f"\n{'='*70}")
        print(f"END-EFFECTOR TRAJECTORY")
        print(f"{'='*70}\n")
        
        print(f"EE position range:")
        print(f"  X: [{all_ee_pos[:, 0].min():.4f}, {all_ee_pos[:, 0].max():.4f}]")
        print(f"  Y: [{all_ee_pos[:, 1].min():.4f}, {all_ee_pos[:, 1].max():.4f}]")
        print(f"  Z: [{all_ee_pos[:, 2].min():.4f}, {all_ee_pos[:, 2].max():.4f}]")
        
        # Check if demonstrations reach goal
        goal_z = 0.25
        final_positions = [demo[-1]['ee_pos'] for demo in expert_demos]
        final_errors = [np.linalg.norm(pos - np.array([0.5, 0.0, goal_z])) for pos in final_positions]
        
        print(f"\nFinal position errors:")
        print(f"  Mean: {np.mean(final_errors)*1000:.2f}mm")
        print(f"  Max: {np.max(final_errors)*1000:.2f}mm")
        print(f"  Min: {np.min(final_errors)*1000:.2f}mm")
        
        success_count = sum(1 for e in final_errors if e < 0.01)
        print(f"\nSuccess rate (< 10mm error): {success_count}/{len(expert_demos)} ({success_count/len(expert_demos)*100:.1f}%)")
        
        if success_count < len(expert_demos) * 0.5:
            print(f"\n⚠️  WARNING: Low success rate in expert demonstrations!")
    
    # Evaluate expert with environment
    print(f"\n{'='*70}")
    print(f"ENVIRONMENT REWARD CHECK")
    print(f"{'='*70}\n")
    
    env = PegInHoleEnv()
    expert_rewards = []
    
    print("Replaying expert demonstrations in environment...")
    for i, demo in enumerate(expert_demos[:3]):  # Check first 3
        state, _ = env.reset()
        episode_reward = 0.0
        
        for trans in demo:
            action = trans['action']
            _, reward, _, _, _ = env.step(action)
            episode_reward += reward
        
        expert_rewards.append(episode_reward)
        print(f"  Demo {i+1}: Total reward = {episode_reward:.2f}")
    
    print(f"\nExpert mean reward: {np.mean(expert_rewards):.2f}")
    
    if np.mean(expert_rewards) < -500:
        print(f"\n⚠️  WARNING: Expert demonstrations have very negative rewards!")
        print(f"   This suggests the expert policy itself may be suboptimal.")
        print(f"   Expected reward should be closer to 0 for good demonstrations.")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"DIAGNOSIS SUMMARY")
    print(f"{'='*70}\n")
    
    issues = []
    
    if np.all(action_std < 1e-6):
        issues.append("✗ Actions are constant (fixed impedance)")
    
    if len(all_forces) > 0 and max_forces.max() > 100:
        issues.append("✗ Very high contact forces detected")
    
    if len(all_ee_pos) > 0 and success_count < len(expert_demos) * 0.5:
        issues.append("✗ Low success rate in demonstrations")
    
    if len(expert_rewards) > 0 and np.mean(expert_rewards) < -500:
        issues.append("✗ Expert demonstrations have very negative rewards")
    
    if len(issues) == 0:
        print("✓ No major issues detected")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        
        print(f"\nRECOMMENDATIONS:")
        if np.all(action_std < 1e-6):
            print("  1. Regenerate expert data with time-varying impedance")
            print("     (modify generate_expert_data.py)")
        if len(expert_rewards) > 0 and np.mean(expert_rewards) < -500:
            print("  2. Check peg_in_hole_env.py reward function")
            print("  3. Verify expert trajectory quality")
    
    env.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        diagnose_expert_data(sys.argv[1])
    else:
        diagnose_expert_data()