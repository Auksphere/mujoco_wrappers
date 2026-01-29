#!/usr/bin/env python3
"""
Adversarial Inverse Reinforcement Learning (AIRL) for Variable Impedance Control

Implementation based on:
"Learning Variable Impedance Control via Inverse Reinforcement Learning for Force-Related Tasks"
Zhang et al., IEEE RA-L 2021

AIRL learns a reward function and policy simultaneously from expert demonstrations,
allowing the agent to learn variable impedance parameters that generalize to new situations.

Key components:
1. Discriminator (reward function): Distinguishes expert from policy trajectories
2. Policy network: Outputs impedance parameters given state
3. Value network: Estimates state values for advantage computation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import pickle
import os
from tqdm import tqdm
from collections import deque
import gymnasium as gym

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from envs.peg_in_hole_env import PegInHoleEnv


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs impedance parameters given state.
    Uses a Gaussian policy for continuous action space.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log std for Gaussian policy
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = torch.sigmoid(self.mean(x))  # Ensure output in [0, 1]
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state):
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0.0, 1.0)  # Ensure valid range
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate(self, state, action):
        """Evaluate log probability of action."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Value network for estimating state values."""
    
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value(x)
        
        return value


class DiscriminatorNetwork(nn.Module):
    """
    AIRL discriminator that learns the reward function.
    
    The discriminator outputs:
    D(s,a,s') = exp(f(s,a,s')) / (exp(f(s,a,s')) + π(a|s))
    
    where f(s,a,s') = r(s,a) + γV(s') - V(s) is the advantage function.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99):
        super(DiscriminatorNetwork, self).__init__()
        
        self.gamma = gamma
        
        # Reward function r(s, a)
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value function V(s)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state, action, next_state):
        """Compute discriminator output."""
        sa = torch.cat([state, action], dim=-1)
        reward = self.reward_net(sa)
        value = self.value_net(state)
        next_value = self.value_net(next_state)
        
        # Advantage: r(s,a) + γV(s') - V(s)
        advantage = reward + self.gamma * next_value - value
        
        return advantage
    
    def get_reward(self, state, action):
        """Get learned reward r(s, a)."""
        sa = torch.cat([state, action], dim=-1)
        reward = self.reward_net(sa)
        return reward


class AIRLTrainer:
    """AIRL training algorithm."""
    
    def __init__(
        self,
        env,
        expert_data,
        state_dim=34,
        action_dim=12,
        hidden_dim=256,
        lr_policy=3e-4,
        lr_discriminator=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_size=100000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.env = env
        self.expert_data = expert_data
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Gradient clipping threshold
        self.max_grad_norm = 0.5
        
        # Print device information
        print(f"\n{'='*60}")
        print(f"Device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"{'='*60}\n")
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.value = ValueNetwork(state_dim, hidden_dim).to(device)
        self.discriminator = DiscriminatorNetwork(state_dim, action_dim, hidden_dim, gamma).to(device)
        
        # Enable cudnn autotuner for better performance
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Optimizers with better learning rates for AIRL stability
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_policy)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_discriminator * 0.5)  # Slower discriminator learning
        
        # Replay buffer for policy samples
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Convert expert data to tensors
        self.expert_states = []
        self.expert_actions = []
        self.expert_next_states = []
        
        for traj in expert_data:
            for transition in traj:
                self.expert_states.append(transition['state'])
                self.expert_actions.append(transition['action'])
                self.expert_next_states.append(transition['next_state'])
        
        self.expert_states = torch.FloatTensor(np.array(self.expert_states)).to(device)
        self.expert_actions = torch.FloatTensor(np.array(self.expert_actions)).to(device)
        self.expert_next_states = torch.FloatTensor(np.array(self.expert_next_states)).to(device)
        
        print(f"Loaded {len(self.expert_states)} expert transitions")
        
    def collect_trajectories(self, n_episodes=10):
        """Collect trajectories using current policy."""
        trajectories = []
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            trajectory = []
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _ = self.policy.sample(state_tensor)
                action_np = action.cpu().numpy()[0]
                
                next_state, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated
                
                trajectory.append({
                    'state': state,
                    'action': action_np,
                    'next_state': next_state,
                    'done': done
                })
                
                state = next_state
            
            trajectories.append(trajectory)
            
            # Add to replay buffer
            for trans in trajectory:
                self.replay_buffer.append(trans)
        
        return trajectories
    
    def update_discriminator(self, n_updates=5):  # Reduce discriminator updates
        """Update discriminator to distinguish expert from policy."""
        discriminator_losses = []
        
        # Balance expert and policy batch sizes
        expert_batch_size = self.batch_size // 2
        policy_batch_size = self.batch_size // 2
        
        for _ in range(n_updates):
            # Sample expert data (already on GPU)
            expert_indices = torch.randint(0, len(self.expert_states), (expert_batch_size,))
            expert_s = self.expert_states[expert_indices]
            expert_a = self.expert_actions[expert_indices]
            expert_ns = self.expert_next_states[expert_indices]
            
            # Sample policy data
            if len(self.replay_buffer) < policy_batch_size:
                continue
            
            # Faster batch sampling
            policy_indices = np.random.choice(len(self.replay_buffer), policy_batch_size, replace=False)
            policy_batch = [self.replay_buffer[i] for i in policy_indices]
            
            # Batch convert to tensors (faster than list comprehension)
            policy_states = np.array([t['state'] for t in policy_batch])
            policy_actions = np.array([t['action'] for t in policy_batch])
            policy_next_states = np.array([t['next_state'] for t in policy_batch])
            
            policy_s = torch.from_numpy(policy_states).float().to(self.device, non_blocking=True)
            policy_a = torch.from_numpy(policy_actions).float().to(self.device, non_blocking=True)
            policy_ns = torch.from_numpy(policy_next_states).float().to(self.device, non_blocking=True)
            
            # Compute discriminator outputs
            expert_adv = self.discriminator(expert_s, expert_a, expert_ns)
            policy_adv = self.discriminator(policy_s, policy_a, policy_ns)
            
            # Use logistic regression loss with label smoothing
            expert_labels = torch.ones_like(expert_adv) * 0.9  # Label smoothing
            policy_labels = torch.zeros_like(policy_adv) + 0.1  # Label smoothing
            
            # AIRL discriminator loss with label smoothing for stability
            expert_loss = F.binary_cross_entropy_with_logits(
                expert_adv, expert_labels, reduction='mean'
            )
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_adv, policy_labels, reduction='mean'
            )
            
            discriminator_loss = expert_loss + policy_loss
            
            # Reduced L2 regularization
            l2_reg = 0.0
            for param in self.discriminator.parameters():
                l2_reg += torch.norm(param)
            discriminator_loss += 1e-5 * l2_reg  # Reduced regularization
            
            # Check for NaN
            if torch.isnan(discriminator_loss):
                print("Warning: NaN detected in discriminator loss, skipping update")
                continue
            
            # Update discriminator with gradient clipping
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()
            
            discriminator_losses.append(discriminator_loss.item())
        
        return np.mean(discriminator_losses) if discriminator_losses else 0.0
    
    def update_policy(self, n_updates=10):
        """Update policy using learned rewards from discriminator."""
        policy_losses = []
        value_losses = []
        
        for _ in range(n_updates):
            if len(self.replay_buffer) < self.batch_size:
                continue
            
            # Sample from replay buffer - optimized
            batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in batch_indices]
            
            # Batch convert to tensors
            batch_states = np.array([t['state'] for t in batch])
            batch_actions = np.array([t['action'] for t in batch])
            batch_next_states = np.array([t['next_state'] for t in batch])
            batch_dones = np.array([t['done'] for t in batch])
            
            states = torch.from_numpy(batch_states).float().to(self.device, non_blocking=True)
            actions = torch.from_numpy(batch_actions).float().to(self.device, non_blocking=True)
            next_states = torch.from_numpy(batch_next_states).float().to(self.device, non_blocking=True)
            dones = torch.from_numpy(batch_dones).float().to(self.device, non_blocking=True)
            
            # Get rewards from discriminator
            with torch.no_grad():
                rewards = self.discriminator.get_reward(states, actions).squeeze()
                rewards = torch.clamp(rewards, -10, 10)  # Clamp rewards
            
            # Compute values
            values = self.value(states).squeeze()
            next_values = self.value(next_states).squeeze()
            
            # Compute advantages with clamping
            advantages = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = torch.clamp(advantages, -10, 10)  # Clamp advantages
            
            # Normalize advantages
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update value network
            target_values = rewards + self.gamma * next_values.detach() * (1 - dones)
            target_values = torch.clamp(target_values, -10, 10)
            value_loss = F.mse_loss(values, target_values)
            
            # Check for NaN
            if torch.isnan(value_loss):
                print("Warning: NaN detected in value loss, skipping update")
                continue
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
            # Update policy network
            log_probs, entropy = self.policy.evaluate(states, actions)
            log_probs = torch.clamp(log_probs, -10, 2)
            
            policy_loss = -(log_probs * advantages.detach()).mean() - 0.01 * entropy.mean()
            
            # Check for NaN
            if torch.isnan(policy_loss):
                print("Warning: NaN detected in policy loss, skipping update")
                continue
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        return np.mean(policy_losses) if policy_losses else 0.0, np.mean(value_losses) if value_losses else 0.0
    
    def train(self, n_epochs=1000, n_episodes_per_epoch=10, save_dir='models/airl', 
              discriminator_updates=100, policy_updates=100, eval_freq=10, save_freq=100):
        """Train AIRL."""
        os.makedirs(save_dir, exist_ok=True)
        
        best_reward = -np.inf
        
        print(f"\nStarting AIRL Training:")
        print(f"  Epochs: {n_epochs}")
        print(f"  Episodes per epoch: {n_episodes_per_epoch}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Discriminator updates: {discriminator_updates}")
        print(f"  Policy updates: {policy_updates}")
        print(f"  Learning rates: policy={self.policy_optimizer.param_groups[0]['lr']:.2e}, "
              f"discriminator={self.discriminator_optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Gradient clipping: {self.max_grad_norm}\n")
        
        for epoch in tqdm(range(n_epochs), desc="Training AIRL"):
            # Collect trajectories
            trajectories = self.collect_trajectories(n_episodes_per_epoch)
            
            # Adaptive update schedule - reduce discriminator updates as training progresses
            current_disc_updates = max(discriminator_updates // (1 + epoch // 100), 5)
            current_policy_updates = policy_updates
            
            # Update discriminator less frequently to prevent overfitting
            disc_loss = self.update_discriminator(n_updates=current_disc_updates)
            
            # Update policy more frequently to catch up
            policy_loss, value_loss = self.update_policy(n_updates=current_policy_updates)
            
            # Evaluate
            if epoch % eval_freq == 0:
                eval_reward = self.evaluate(n_episodes=3)  # Reduce eval episodes for speed
                
                print(f"\nEpoch {epoch}/{n_epochs}")
                print(f"  Discriminator Loss: {disc_loss:.4f}")
                print(f"  Policy Loss: {policy_loss:.4f}")
                print(f"  Value Loss: {value_loss:.4f}")
                print(f"  Eval Reward: {eval_reward:.4f}")
                print(f"  Replay Buffer Size: {len(self.replay_buffer)}")
                print(f"  Discriminator Updates: {current_disc_updates}")
                
                # Save best model
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self.save(os.path.join(save_dir, 'best_model.pt'))
                    print(f"  Saved best model (reward: {best_reward:.4f})")
                    
                # Early stopping if discriminator is too confident
                if disc_loss < 0.01 and epoch > 50:
                    print(f"  Warning: Discriminator loss very low ({disc_loss:.4f}), may need rebalancing")
            
            # Save checkpoint
            if epoch % save_freq == 0 and epoch > 0:
                self.save(os.path.join(save_dir, f'checkpoint_{epoch}.pt'))
        
        # Save final model
        self.save(os.path.join(save_dir, 'final_model.pt'))
        print("\nTraining completed!")
    
    def evaluate(self, n_episodes=10):
        """Evaluate policy."""
        total_reward = 0.0
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _ = self.policy.sample(state_tensor)
                action_np = action.cpu().numpy()[0]
                
                state, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / n_episodes
    
    def save(self, path):
        """Save model."""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import AIRL_CONFIG, ENV_CONFIG
    
    # Load expert data (use optimized version if available)
    expert_path = 'data/expert_demonstrations_optimized.pkl'
    if not os.path.exists(expert_path):
        expert_path = 'data/expert_demonstrations.pkl'
    
    with open(expert_path, 'rb') as f:
        expert_data = pickle.load(f)
    
    print(f"Loaded {len(expert_data)} expert demonstrations from {expert_path}")
    
    # Create environment
    env = PegInHoleEnv()
    
    # Create trainer with config
    trainer = AIRLTrainer(
        env=env,
        expert_data=expert_data,
        state_dim=AIRL_CONFIG['state_dim'],
        action_dim=AIRL_CONFIG['action_dim'],
        hidden_dim=AIRL_CONFIG['hidden_dim'],
        lr_policy=AIRL_CONFIG['lr_policy'],
        lr_discriminator=AIRL_CONFIG['lr_discriminator'],
        batch_size=AIRL_CONFIG['batch_size'],
        gamma=AIRL_CONFIG['gamma'],
        tau=AIRL_CONFIG['tau'],
        buffer_size=AIRL_CONFIG['buffer_size']
    )
    
    # Train with config parameters
    trainer.train(
        n_epochs=AIRL_CONFIG['n_epochs'],
        n_episodes_per_epoch=AIRL_CONFIG['n_episodes_per_epoch'],
        save_dir=AIRL_CONFIG['save_dir'],
        discriminator_updates=AIRL_CONFIG['discriminator_updates'],
        policy_updates=AIRL_CONFIG['policy_updates'],
        eval_freq=AIRL_CONFIG['eval_freq'],
        save_freq=AIRL_CONFIG['save_freq']
    )
