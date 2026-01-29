#!/usr/bin/env python3
"""
Restart AIRL training with improved parameters

This script addresses the issues found in the previous training:
1. Discriminator overfitting (loss too low)
2. Policy not learning effectively 
3. Reward stuck at -50.0
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

def main():
    print("🔄 RESTARTING AIRL TRAINING WITH IMPROVED PARAMETERS")
    print("="*70)
    
    # Load expert data
    expert_path = 'data/expert_demonstrations_optimized.pkl'
    if not os.path.exists(expert_path):
        expert_path = 'data/expert_demonstrations.pkl'
    
    with open(expert_path, 'rb') as f:
        expert_data = pickle.load(f)
    
    print(f"✅ Loaded {len(expert_data)} expert demonstrations from {expert_path}")
    
    # Create environment
    env = PegInHoleEnv()
    print("✅ Environment created")
    
    # Improved training parameters
    improved_config = {
        'state_dim': 34,
        'action_dim': 12, 
        'hidden_dim': 256,
        'lr_policy': 1e-4,        # Higher for policy
        'lr_discriminator': 3e-5,  # Much lower for discriminator
        'batch_size': 256,         # Smaller batch for better learning
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_size': 50000,      # Smaller buffer
    }
    
    # Create trainer with improved config
    trainer = AIRLTrainer(
        env=env,
        expert_data=expert_data,
        **improved_config
    )
    
    print("✅ AIRL Trainer created with improved parameters")
    print(f"   Policy LR: {improved_config['lr_policy']:.2e}")
    print(f"   Discriminator LR: {improved_config['lr_discriminator']:.2e}")
    print(f"   Batch size: {improved_config['batch_size']}")
    
    # Training parameters
    training_config = {
        'n_epochs': 500,           # Reduced for faster iteration
        'n_episodes_per_epoch': 8, # Reduced episodes per epoch
        'save_dir': 'script/models/airl_improved',
        'discriminator_updates': 5, # Much fewer discriminator updates
        'policy_updates': 20,       # More policy updates
        'eval_freq': 10,
        'save_freq': 50,
    }
    
    print(f"✅ Training configuration:")
    print(f"   Epochs: {training_config['n_epochs']}")
    print(f"   Episodes/epoch: {training_config['n_episodes_per_epoch']}")
    print(f"   Discriminator updates: {training_config['discriminator_updates']}")
    print(f"   Policy updates: {training_config['policy_updates']}")
    
    # Start training
    print("\n🚀 Starting improved AIRL training...")
    try:
        trainer.train(**training_config)
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        # Save current state
        trainer.save('script/models/airl_improved/interrupted_model.pt')
        print("💾 Model saved before exit")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💾 Attempting to save current state...")
        try:
            trainer.save('script/models/airl_improved/error_model.pt')
            print("✅ Model saved successfully")
        except:
            print("❌ Failed to save model")

if __name__ == "__main__":
    main()