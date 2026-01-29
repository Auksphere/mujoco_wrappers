#!/usr/bin/env python3
"""
Complete Workflow Script for Variable Impedance Control Learning

This script runs the complete pipeline:
1. Generate expert demonstrations
2. Train AIRL policy
3. Evaluate trained policy
"""

import os
import sys
import argparse
import subprocess


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError: {description} failed with code {result.returncode}")
        sys.exit(1)
    
    print(f"\n{description} completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Run complete AIRL workflow')
    parser.add_argument('--skip-expert', action='store_true',
                        help='Skip expert data generation (use existing data)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training (use existing model)')
    parser.add_argument('--n-demos', type=int, default=10,
                        help='Number of expert demonstrations')
    parser.add_argument('--n-epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--n-eval', type=int, default=10,
                        help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\n" + "="*60)
    print("Variable Impedance Control Learning via AIRL")
    print("="*60)
    
    # Step 1: Generate expert demonstrations
    if not args.skip_expert:
        print(f"\nStep 1: Generating {args.n_demos} expert demonstrations...")
        
        # Check if expert data already exists
        if os.path.exists('../../data/expert_demonstrations.pkl'):
            response = input("\nExpert data already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Skipping expert data generation.")
                args.skip_expert = True
        
        if not args.skip_expert:
            cmd = f"python generate_expert_data.py --n-demos {args.n_demos}"
            run_command(cmd, "Expert Data Generation")
    else:
        print("\nStep 1: Skipping expert data generation (using existing data)")
        
        # Check if data exists
        if not os.path.exists('../../data/expert_demonstrations.pkl'):
            print("\nError: No expert data found. Please generate it first.")
            sys.exit(1)
    
    # Step 2: Train AIRL policy
    if not args.skip_train:
        print(f"\nStep 2: Training AIRL policy for {args.n_epochs} epochs...")
        
        # Check if model already exists
        if os.path.exists('../../models/airl/best_model.pt'):
            response = input("\nModel already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Skipping training.")
                args.skip_train = True
        
        if not args.skip_train:
            cmd = f"python train_airl.py --n-epochs {args.n_epochs}"
            run_command(cmd, "AIRL Training")
    else:
        print("\nStep 2: Skipping training (using existing model)")
        
        # Check if model exists
        if not os.path.exists('../../models/airl/best_model.pt'):
            print("\nError: No trained model found. Please train first.")
            sys.exit(1)
    
    # Step 3: Evaluate policy
    print(f"\nStep 3: Evaluating trained policy ({args.n_eval} episodes)...")
    cmd = f"python evaluate_policy.py --model ../../models/airl/best_model.pt --n-episodes {args.n_eval}"
    run_command(cmd, "Policy Evaluation")
    
    # Summary
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  - Expert data: data/expert_demonstrations.pkl")
    print("  - Trained model: models/airl/best_model.pt")
    print("  - Visualizations: plots/airl_trajectory.png")
    print("\nTo run a single episode with rendering:")
    print("  python evaluate_policy.py --model ../../models/airl/best_model.pt --render")
    print("\n")


if __name__ == "__main__":
    main()
