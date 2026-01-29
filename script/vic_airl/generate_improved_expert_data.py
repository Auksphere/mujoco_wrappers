#!/usr/bin/env python3
"""
Generate improved expert data using the GIC-based expert strategy.

This script generates expert demonstrations with:
1. GIC-style adaptive impedance control
2. Improved reward function
3. Better action space utilization
"""

import os
import sys
import pickle
import argparse
import numpy as np

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from script.vic_airl.generate_expert_data import ExpertDemonstrationGenerator
from envs.peg_in_hole_env import PegInHoleEnv


def validate_expert_data(demo_data):
    """Validate the generated expert data quality."""
    print("\n" + "="*60)
    print("EXPERT DATA VALIDATION")
    print("="*60)
    
    # Basic statistics
    n_demos = len(demo_data)
    trajectory_lengths = [len(demo) for demo in demo_data]
    
    print(f"Number of demonstrations: {n_demos}")
    print(f"Trajectory lengths: min={min(trajectory_lengths)}, max={max(trajectory_lengths)}, avg={np.mean(trajectory_lengths):.1f}")
    
    # Analyze actions (impedance parameters)
    all_actions = []
    for demo in demo_data:
        for transition in demo:
            all_actions.append(transition['action'])
    
    all_actions = np.array(all_actions)
    action_means = np.mean(all_actions, axis=0)
    action_stds = np.std(all_actions, axis=0)
    
    print(f"\nAction analysis:")
    print(f"  Action shape: {all_actions.shape}")
    print(f"  Linear Kp range: [{action_means[0]:.3f}±{action_stds[0]:.3f}, {action_means[1]:.3f}±{action_stds[1]:.3f}, {action_means[2]:.3f}±{action_stds[2]:.3f}]")
    print(f"  Angular Kp range: [{action_means[6]:.3f}±{action_stds[6]:.3f}, {action_means[7]:.3f}±{action_stds[7]:.3f}, {action_means[8]:.3f}±{action_stds[8]:.3f}]")
    
    if np.all(action_stds > 1e-6):
        print("  ✓ Actions show good variation")
    else:
        print("  ⚠️ Some actions are constant (low variation)")
    
    # Analyze forces
    all_forces = []
    for demo in demo_data:
        for transition in demo:
            if 'force' in transition:
                all_forces.append(np.linalg.norm(transition['force']))
    
    if all_forces:
        max_force = np.max(all_forces)
        avg_force = np.mean(all_forces)
        print(f"\nForce analysis:")
        print(f"  Max force: {max_force:.2f}N")
        print(f"  Average force: {avg_force:.2f}N")
        
        if max_force < 100:  # Reasonable force range
            print("  ✓ Forces are within reasonable range")
        else:
            print("  ⚠️ Very high forces detected")
    
    # Analyze final positions (success rate)
    final_errors = []
    for demo in demo_data:
        if len(demo) > 0:
            final_pos = demo[-1]['ee_pos']

            goal_pos = np.array([0.0, -0.7, 0.11])  # 1cm above hole
            error = np.linalg.norm(final_pos - goal_pos)
            final_errors.append(error)
    
    if final_errors:
        success_count = sum(1 for e in final_errors if e < 0.03)  # 3cm threshold
        success_rate = success_count / len(final_errors)
        
        print(f"\nSuccess analysis:")
        print(f"  Final position errors: min={np.min(final_errors)*1000:.1f}mm, max={np.max(final_errors)*1000:.1f}mm, avg={np.mean(final_errors)*1000:.1f}mm")
        print(f"  Success rate (< 30mm): {success_rate*100:.1f}% ({success_count}/{len(final_errors)})")
        
        if success_rate > 0.7:
            print("  ✓ Good success rate")
        elif success_rate > 0.3:
            print("  ⚠️ Moderate success rate")
        else:
            print("  ✗ Low success rate")


def test_rewards_on_expert_data(demo_data):
    """Test the improved reward function on expert data."""
    print("\n" + "="*60)
    print("REWARD FUNCTION TESTING")
    print("="*60)
    
    env = PegInHoleEnv()
    
    all_rewards = []
    for i, demo in enumerate(demo_data[:3]):  # Test first 3 demos
        print(f"\nTesting demo {i+1}:")
        
        demo_rewards = []
        for step, transition in enumerate(demo):
            try:
                reward = env._compute_reward(transition['state'], transition['action'])
                demo_rewards.append(reward)
                
                # Print some sample rewards
                if step % (len(demo) // 5) == 0:  # Print 5 samples per demo
                    print(f"  Step {step}/{len(demo)}: reward = {reward:.3f}")
                    
            except Exception as e:
                print(f"  Warning: Could not compute reward at step {step}: {e}")
                continue
        
        if demo_rewards:
            total_reward = sum(demo_rewards)
            avg_reward = np.mean(demo_rewards)
            max_reward = np.max(demo_rewards)
            
            all_rewards.extend(demo_rewards)
            
            print(f"  Demo {i+1} summary:")
            print(f"    Total reward: {total_reward:.2f}")
            print(f"    Average reward: {avg_reward:.3f}")
            print(f"    Max reward: {max_reward:.3f}")
            
            if avg_reward > -1.0:
                print(f"    ✓ Reasonable rewards (not too negative)")
            else:
                print(f"    ⚠️ Very negative rewards")
    
    if all_rewards:
        print(f"\nOverall reward statistics:")
        print(f"  Mean reward: {np.mean(all_rewards):.3f}")
        print(f"  Std reward: {np.std(all_rewards):.3f}")
        print(f"  Min reward: {np.min(all_rewards):.3f}")
        print(f"  Max reward: {np.max(all_rewards):.3f}")
        
        if np.mean(all_rewards) > -2.0:
            print("  ✓ Expert rewards are reasonable for AIRL training")
        else:
            print("  ⚠️ Expert rewards may be too negative for effective AIRL training")


def generate_improved_expert_data(n_demos=10, save_path="../../data/expert_demonstrations_improved.pkl"):
    """Generate expert data using the improved implementation."""
    print("="*60)
    print("GENERATING IMPROVED EXPERT DATA")
    print("="*60)
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create expert generator
    generator = ExpertDemonstrationGenerator(
        xml_path="models/jaka_zu12/jaka_pih.xml",
        control_dt=0.04
    )
    
    print(f"Generating {n_demos} expert demonstrations...")
    print("Using improved GIC-based expert strategy")
    
    # Generate demonstrations
    demo_data = []
    for i in range(n_demos):
        print(f"\n--- Generating demonstration {i+1}/{n_demos} ---")
        try:
            demo = generator.generate_demonstration(verbose=True)
            demo_data.append(demo)
            print(f"✓ Demo {i+1} completed with {len(demo)} transitions")
        except Exception as e:
            print(f"✗ Demo {i+1} failed: {e}")
            continue
    
    if len(demo_data) == 0:
        print("✗ No demonstrations were generated successfully!")
        return None
    
    # Validate the generated data
    validate_expert_data(demo_data)
    
    # Test rewards
    test_rewards_on_expert_data(demo_data)
    
    # Save the data
    with open(save_path, 'wb') as f:
        pickle.dump(demo_data, f)
    
    print(f"\n✓ Expert data saved to: {save_path}")
    print(f"  Generated {len(demo_data)}/{n_demos} demonstrations")
    print(f"  Total transitions: {sum(len(demo) for demo in demo_data)}")
    
    return demo_data


def main():
    parser = argparse.ArgumentParser(description='Generate improved expert data for AIRL training')
    parser.add_argument('--n-demos', type=int, default=10,
                        help='Number of expert demonstrations to generate')
    parser.add_argument('--save-path', type=str, default="../../data/expert_demonstrations_improved.pkl",
                        help='Path to save expert data')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test the implementation without generating full dataset')
    
    args = parser.parse_args()
    
    if args.test_only:
        print("Running test mode (1 demo only)...")
        demo_data = generate_improved_expert_data(n_demos=1, save_path=args.save_path.replace('.pkl', '_test.pkl'))
    else:
        demo_data = generate_improved_expert_data(n_demos=args.n_demos, save_path=args.save_path)
    
    if demo_data is not None:
        print("\n" + "="*60)
        print("SUMMARY OF IMPROVEMENTS")
        print("="*60)
        print("✓ GIC-style adaptive impedance control")
        print("✓ Multi-stage reward function")
        print("✓ Task-phase aware expert strategy")
        print("✓ Proper action space utilization")
        print("\nThe generated expert data should now provide reasonable rewards")
        print("for AIRL training, addressing the original issue of too-negative rewards.")


if __name__ == "__main__":
    main()