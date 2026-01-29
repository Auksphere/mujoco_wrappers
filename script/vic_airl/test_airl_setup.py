#!/usr/bin/env python3
"""
Quick test script to verify AIRL improvements before full training
"""

import os
import sys
import pickle
import numpy as np
import torch

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from envs.peg_in_hole_env import PegInHoleEnv
from train_airl import AIRLTrainer

def test_airl_setup():
    print("🧪 TESTING IMPROVED AIRL SETUP")
    print("="*50)
    
    # Load expert data
    expert_path = 'data/expert_demonstrations_optimized.pkl'
    with open(expert_path, 'rb') as f:
        expert_data = pickle.load(f)
    
    print(f"✅ Expert data loaded: {len(expert_data)} demonstrations")
    
    # Create environment  
    env = PegInHoleEnv()
    print("✅ Environment created")
    
    # Test config
    config = {
        'state_dim': 34,
        'action_dim': 12,
        'hidden_dim': 128,  # Smaller for testing
        'lr_policy': 1e-4,
        'lr_discriminator': 3e-5,
        'batch_size': 64,   # Small batch for testing
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_size': 1000,
    }
    
    # Create trainer
    trainer = AIRLTrainer(env=env, expert_data=expert_data, **config)
    print("✅ Trainer created")
    
    # Test data loading
    print(f"Expert transitions: {len(trainer.expert_states)}")
    print(f"Expert action shape: {trainer.expert_actions.shape}")
    print(f"Expert action stats: mean={trainer.expert_actions.mean():.3f}, std={trainer.expert_actions.std():.3f}")
    
    # Test trajectory collection
    print("\n🔄 Testing trajectory collection...")
    trajectories = trainer.collect_trajectories(n_episodes=2)
    print(f"Collected {len(trajectories)} trajectories")
    
    if len(trajectories) > 0:
        traj_len = len(trajectories[0])
        print(f"First trajectory length: {traj_len}")
        
        # Test action diversity
        actions = [step['action'] for step in trajectories[0]]
        actions = np.array(actions)
        print(f"Policy action stats: mean={actions.mean():.3f}, std={actions.std():.3f}")
    
    # Test discriminator update
    print("\n🔄 Testing discriminator update...")
    if len(trainer.replay_buffer) >= config['batch_size']:
        disc_loss = trainer.update_discriminator(n_updates=2)
        print(f"Discriminator loss: {disc_loss:.4f}")
    else:
        print("Not enough data in replay buffer for discriminator update")
    
    # Test policy update
    print("\n🔄 Testing policy update...")
    if len(trainer.replay_buffer) >= config['batch_size']:
        policy_loss, value_loss = trainer.update_policy(n_updates=2)
        print(f"Policy loss: {policy_loss:.4f}")
        print(f"Value loss: {value_loss:.4f}")
    else:
        print("Not enough data in replay buffer for policy update")
    
    # Test evaluation
    print("\n🔄 Testing evaluation...")
    eval_reward = trainer.evaluate(n_episodes=1)
    print(f"Evaluation reward: {eval_reward:.4f}")
    
    print("\n✅ All tests completed successfully!")
    
    # Check for obvious issues
    issues = []
    if len(trainer.expert_states) < 100:
        issues.append("Very few expert transitions")
    if trainer.expert_actions.std() < 0.01:
        issues.append("Expert actions have very low diversity")
    if eval_reward < -100:
        issues.append("Evaluation reward is very negative")
    
    if issues:
        print("\n⚠️ Potential issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n🎉 No obvious issues detected - ready for training!")

if __name__ == "__main__":
    test_airl_setup()