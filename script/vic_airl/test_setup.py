#!/usr/bin/env python3
"""
Quick test script to verify environment setup

Tests:
1. Import all required modules
2. Create and reset Peg-in-Hole environment
3. Run a few random steps
4. Check observation and action spaces
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    try:
        import numpy as np
        import mujoco
        import gymnasium as gym
        from scipy.spatial.transform import Rotation
        print("✓ All basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_environment():
    """Test Peg-in-Hole environment."""
    print("\nTesting Peg-in-Hole environment...")
    try:
        from envs.peg_in_hole_env import PegInHoleEnv
        
        # Create environment
        env = PegInHoleEnv(xml_path="models/jaka_zu12/jaka_pih.xml")
        print(f"✓ Environment created successfully")
        
        # Check spaces
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        
        # Reset environment
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Initial observation shape: {obs.shape}")
        
        # Run a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.3f}, done={terminated or truncated}")
        
        print("✓ Environment test passed")
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_expert_generator():
    """Test expert demonstration generator."""
    print("\nTesting expert demonstration generator...")
    try:
        from script.vic_airl.generate_expert_data import ExpertDemonstrationGenerator
        
        # Create generator
        generator = ExpertDemonstrationGenerator(
            xml_path="models/jaka_zu12/jaka_pih.xml"
        )
        print("✓ Expert generator created")
        
        # Note: We don't run full generation here as it takes time
        print("  (Skipping full generation - use generate_expert_data.py for that)")
        
        return True
        
    except Exception as e:
        print(f"✗ Expert generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch():
    """Test PyTorch installation (needed for AIRL training)."""
    print("\nTesting PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("AIRL Variable Impedance Control - Setup Verification")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("Expert Generator", test_expert_generator()))
    results.append(("PyTorch", test_pytorch()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run workflow.")
        print("\nNext steps:")
        print("  1. Generate expert data: python generate_expert_data.py")
        print("  2. Train AIRL: python train_airl.py")
        print("  3. Evaluate: python evaluate_policy.py")
        print("\nOr run complete workflow: python run_workflow.py")
    else:
        print("\n✗ Some tests failed. Please fix issues before proceeding.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
