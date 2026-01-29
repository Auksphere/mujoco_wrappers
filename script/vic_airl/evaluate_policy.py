#!/usr/bin/env python3
"""
Test and Evaluate Trained AIRL Policy

This script loads a trained AIRL policy and evaluates it on the peg-in-hole task.
It visualizes the learned impedance parameters and compares performance with expert.
"""

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from envs.peg_in_hole_env import PegInHoleEnv
from script.vic_airl.train_airl import PolicyNetwork


class AIRLEvaluator:
    """Evaluate trained AIRL policy."""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load environment
        self.env = PegInHoleEnv(render_mode=None)
        
        # Load policy
        self.policy = PolicyNetwork(state_dim=34, action_dim=12, hidden_dim=256).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy.eval()
        
        print(f"Loaded policy from {model_path}")
    
    def evaluate_episode(self, render=False, max_steps=500):
        """Evaluate one episode and return trajectory data."""
        state, _ = self.env.reset()
        done = False
        step = 0
        
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'impedance_params': [],
            'forces': [],
            'ee_positions': [],
            'success': False
        }
        
        while not done and step < max_steps:
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.policy.sample(state_tensor)
            action_np = action.cpu().numpy()[0]
            
            # Unpack impedance parameters
            kp_linear = action_np[0:3] * (2000.0 - 100.0) + 100.0
            kd_linear = action_np[3:6] * (200.0 - 10.0) + 10.0
            kp_angular = action_np[6:9] * (500.0 - 50.0) + 50.0
            kd_angular = action_np[9:12] * (100.0 - 5.0) + 5.0
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Extract data from state
            ee_pos = state[12:15]
            force = state[25:28]
            
            # Store trajectory data
            trajectory['states'].append(state)
            trajectory['actions'].append(action_np)
            trajectory['rewards'].append(reward)
            trajectory['impedance_params'].append({
                'kp_linear': kp_linear,
                'kd_linear': kd_linear,
                'kp_angular': kp_angular,
                'kd_angular': kd_angular
            })
            trajectory['forces'].append(force)
            trajectory['ee_positions'].append(ee_pos)
            
            if info.get('is_success', False):
                trajectory['success'] = True
            
            state = next_state
            step += 1
            
            if render:
                self.env.render()
        
        return trajectory
    
    def evaluate_multiple_episodes(self, n_episodes=10):
        """Evaluate multiple episodes and compute statistics."""
        results = {
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'avg_max_force': 0.0,
            'avg_final_error': 0.0,
            'trajectories': []
        }
        
        successes = 0
        total_reward = 0.0
        max_forces = []
        final_errors = []
        
        print(f"\nEvaluating {n_episodes} episodes...")
        for i in range(n_episodes):
            traj = self.evaluate_episode()
            
            if traj['success']:
                successes += 1
            
            episode_reward = sum(traj['rewards'])
            total_reward += episode_reward
            
            forces = np.array(traj['forces'])
            max_force = np.max(np.linalg.norm(forces, axis=1))
            max_forces.append(max_force)
            
            final_pos = traj['ee_positions'][-1]
            goal_pos = np.array([0.5, 0.0, 0.25])
            final_error = np.linalg.norm(final_pos - goal_pos)
            final_errors.append(final_error)
            
            results['trajectories'].append(traj)
            
            print(f"Episode {i+1}/{n_episodes}: Success={traj['success']}, "
                  f"Reward={episode_reward:.2f}, MaxForce={max_force:.2f}N, "
                  f"FinalError={final_error*1000:.2f}mm")
        
        results['success_rate'] = successes / n_episodes
        results['avg_reward'] = total_reward / n_episodes
        results['avg_max_force'] = np.mean(max_forces)
        results['avg_final_error'] = np.mean(final_errors)
        
        print(f"\n--- Evaluation Summary ---")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print(f"Avg Reward: {results['avg_reward']:.2f}")
        print(f"Avg Max Force: {results['avg_max_force']:.2f}N")
        print(f"Avg Final Error: {results['avg_final_error']*1000:.2f}mm")
        
        return results
    
    def visualize_trajectory(self, trajectory, save_path=None):
        """Visualize trajectory data."""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig)
        
        # Time steps
        n_steps = len(trajectory['states'])
        time_steps = np.arange(n_steps) * 0.04  # 25Hz control
        
        # Plot 1: End-effector position
        ax1 = fig.add_subplot(gs[0, :])
        ee_positions = np.array(trajectory['ee_positions'])
        ax1.plot(time_steps, ee_positions[:, 0], label='X')
        ax1.plot(time_steps, ee_positions[:, 1], label='Y')
        ax1.plot(time_steps, ee_positions[:, 2], label='Z')
        ax1.axhline(y=0.25, color='r', linestyle='--', label='Goal Z')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('End-Effector Position')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Forces
        ax2 = fig.add_subplot(gs[1, :])
        forces = np.array(trajectory['forces'])
        ax2.plot(time_steps, forces[:, 0], label='Fx')
        ax2.plot(time_steps, forces[:, 1], label='Fy')
        ax2.plot(time_steps, forces[:, 2], label='Fz')
        ax2.plot(time_steps, np.linalg.norm(forces, axis=1), 'k--', label='|F|')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Force (N)')
        ax2.set_title('Contact Forces')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Linear stiffness
        ax3 = fig.add_subplot(gs[2, 0])
        kp_linear = np.array([p['kp_linear'] for p in trajectory['impedance_params']])
        ax3.plot(time_steps, kp_linear[:, 0], label='Kx')
        ax3.plot(time_steps, kp_linear[:, 1], label='Ky')
        ax3.plot(time_steps, kp_linear[:, 2], label='Kz')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Stiffness (N/m)')
        ax3.set_title('Linear Stiffness (Kp)')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Linear damping
        ax4 = fig.add_subplot(gs[2, 1])
        kd_linear = np.array([p['kd_linear'] for p in trajectory['impedance_params']])
        ax4.plot(time_steps, kd_linear[:, 0], label='Dx')
        ax4.plot(time_steps, kd_linear[:, 1], label='Dy')
        ax4.plot(time_steps, kd_linear[:, 2], label='Dz')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Damping (Ns/m)')
        ax4.set_title('Linear Damping (Kd)')
        ax4.legend()
        ax4.grid(True)
        
        # Plot 5: Rewards
        ax5 = fig.add_subplot(gs[2, 2])
        rewards = trajectory['rewards']
        ax5.plot(time_steps, rewards)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Reward')
        ax5.set_title('Reward Signal')
        ax5.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def compare_with_expert(self, expert_data_path):
        """Compare learned policy with expert demonstrations."""
        # Load expert data
        with open(expert_data_path, 'rb') as f:
            expert_demos = pickle.load(f)
        
        # Evaluate policy
        policy_results = self.evaluate_multiple_episodes(n_episodes=10)
        
        # Compute expert statistics
        expert_forces = []
        expert_errors = []
        
        for demo in expert_demos[:10]:  # Use first 10 demos
            forces = [np.linalg.norm(t['force']) for t in demo]
            expert_forces.append(np.max(forces))
            
            final_pos = demo[-1]['ee_pos']
            goal_pos = np.array([0.5, 0.0, 0.25])
            expert_errors.append(np.linalg.norm(final_pos - goal_pos))
        
        # Print comparison
        print("\n--- Comparison with Expert ---")
        print(f"Expert Avg Max Force: {np.mean(expert_forces):.2f}N")
        print(f"Policy Avg Max Force: {policy_results['avg_max_force']:.2f}N")
        print(f"\nExpert Avg Final Error: {np.mean(expert_errors)*1000:.2f}mm")
        print(f"Policy Avg Final Error: {policy_results['avg_final_error']*1000:.2f}mm")
        print(f"\nPolicy Success Rate: {policy_results['success_rate']*100:.1f}%")
        
        # Visualize one trajectory
        print("\nVisualizing policy trajectory...")
        self.visualize_trajectory(policy_results['trajectories'][0], 
                                   save_path='plots/airl_trajectory.png')


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate AIRL policy')
    parser.add_argument('--model', type=str, default='models/airl/best_model.pt',
                        help='Path to trained model')
    parser.add_argument('--expert', type=str, default='data/expert_demonstrations.pkl',
                        help='Path to expert demonstrations')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render during evaluation')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = AIRLEvaluator(args.model)
    
    if args.render:
        # Evaluate single episode with rendering
        print("Running single episode with rendering...")
        traj = evaluator.evaluate_episode(render=True)
        evaluator.visualize_trajectory(traj)
    else:
        # Evaluate multiple episodes
        evaluator.evaluate_multiple_episodes(n_episodes=args.n_episodes)
        
        # Compare with expert
        if os.path.exists(args.expert):
            evaluator.compare_with_expert(args.expert)


if __name__ == "__main__":
    main()
