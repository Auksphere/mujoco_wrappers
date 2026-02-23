#!/usr/bin/env python3
"""
Plot comparison between expert and policy trajectories
Generate 3 plots:
1. distance_to_hole vs stiffness_norm
2. distance_to_hole vs damping_ratio
3. distance_to_hole vs stiffness_Kz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data():
    """Load expert and policy trajectory data"""
    script_dir = Path(__file__).parent.parent.parent
    log_dir = script_dir / "log"
    
    expert_file = log_dir / "trajectory_pih_expert.csv"
    policy_file = log_dir / "trajectory_pih_policy.csv"
    
    # Ignore rows 2-7 (1-indexed, excluding header) to drop early transient samples.
    skip_transient_rows = list(range(1, 7))

    try:
        expert_data = pd.read_csv(expert_file, skiprows=skip_transient_rows)
        policy_data = pd.read_csv(policy_file, skiprows=skip_transient_rows)
        print(f"Expert data shape: {expert_data.shape}")
        print(f"Policy data shape: {policy_data.shape}")
        return expert_data, policy_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def plot_stiffness_comparison(expert_data, policy_data, save_path=None):
    """Plot distance_to_hole vs stiffness_norm comparison"""
    plt.figure(figsize=(12, 8))
    
    # Plot expert trajectory
    plt.plot(expert_data['distance_to_hole'], expert_data['stiffness_norm'], 
             'b-', linewidth=2, label='Expert', alpha=0.8)
    
    # Plot policy trajectory
    plt.plot(policy_data['distance_to_hole'], policy_data['stiffness_norm'], 
             'r--', linewidth=2, label='Policy', alpha=0.8)
    
    plt.xlabel('Distance to Hole', fontsize=14)
    plt.ylabel('Stiffness Norm', fontsize=14)
    plt.title('Distance to Hole vs Stiffness Norm Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistical information
    expert_mean = expert_data['stiffness_norm'].mean()
    policy_mean = policy_data['stiffness_norm'].mean()
    plt.text(0.02, 0.98, f'Expert Mean: {expert_mean:.1f}\nPolicy Mean: {policy_mean:.1f}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stiffness comparison plot saved to: {save_path}")
    
    plt.show()

def plot_damping_comparison(expert_data, policy_data, save_path=None):
    """Plot distance_to_hole vs damping_ratio comparison"""
    plt.figure(figsize=(12, 8))
    
    # Plot expert trajectory
    plt.plot(expert_data['distance_to_hole'], expert_data['damping_ratio'], 
             'b-', linewidth=2, label='Expert', alpha=0.8)
    
    # Plot policy trajectory
    plt.plot(policy_data['distance_to_hole'], policy_data['damping_ratio'], 
             'r--', linewidth=2, label='Policy', alpha=0.8)
    
    plt.xlabel('Distance to Hole', fontsize=14)
    plt.ylabel('Damping Ratio', fontsize=14)
    plt.title('Distance to Hole vs Damping Ratio Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistical information
    expert_mean = expert_data['damping_ratio'].mean()
    policy_mean = policy_data['damping_ratio'].mean()
    expert_std = expert_data['damping_ratio'].std()
    policy_std = policy_data['damping_ratio'].std()
    
    plt.text(0.02, 0.98, 
             f'Expert: μ={expert_mean:.3f}, σ={expert_std:.3f}\nPolicy: μ={policy_mean:.3f}, σ={policy_std:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Damping comparison plot saved to: {save_path}")
    
    plt.show()


def plot_Kz_comparison(expert_data, policy_data, save_path=None):
    """Plot distance_to_hole vs Kz (3rd position stiffness) comparison"""
    plt.figure(figsize=(12, 8))

    # Plot expert trajectory
    plt.plot(expert_data['distance_to_hole'], expert_data['K3'], 
             'b-', linewidth=2, label='Expert', alpha=0.8)
    
    # Plot policy trajectory
    plt.plot(policy_data['distance_to_hole'], policy_data['K3'], 
             'r--', linewidth=2, label='Policy', alpha=0.8)
    
    plt.xlabel('Distance to Hole', fontsize=14)
    plt.ylabel('Kz (position stiffness)', fontsize=14)
    plt.title('Distance to Hole vs Kz Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistical information
    expert_mean = expert_data['K3'].mean()
    policy_mean = policy_data['K3'].mean()
    plt.text(0.02, 0.98, f'Expert Mean: {expert_mean:.1f}\nPolicy Mean: {policy_mean:.1f}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kz comparison plot saved to: {save_path}")
    
    plt.show()

def analyze_data(expert_data, policy_data):
    """Analyze basic statistics of the data"""
    print("\n=== Data Analysis ===")
    print(f"\nExpert Trajectory:")
    print(f"  Distance to hole: [{expert_data['distance_to_hole'].min():.3f}, {expert_data['distance_to_hole'].max():.3f}]")
    print(f"  Stiffness norm: [{expert_data['stiffness_norm'].min():.1f}, {expert_data['stiffness_norm'].max():.1f}]")
    print(f"  Damping ratio: [{expert_data['damping_ratio'].min():.3f}, {expert_data['damping_ratio'].max():.3f}]")
    print(f"  K3 (position z-axis stiffness): [{expert_data['K3'].min():.1f}, {expert_data['K3'].max():.1f}]")
    
    print(f"\nPolicy Trajectory:")
    print(f"  Distance to hole: [{policy_data['distance_to_hole'].min():.3f}, {policy_data['distance_to_hole'].max():.3f}]")
    print(f"  Stiffness norm: [{policy_data['stiffness_norm'].min():.1f}, {policy_data['stiffness_norm'].max():.1f}]")
    print(f"  Damping ratio: [{policy_data['damping_ratio'].min():.3f}, {policy_data['damping_ratio'].max():.3f}]")
    print(f"  K3 (position z-axis stiffness): [{policy_data['K3'].min():.1f}, {policy_data['K3'].max():.1f}]")

def main():
    """Main function"""
    print("Loading trajectory data...")
    expert_data, policy_data = load_data()
    
    if expert_data is None or policy_data is None:
        print("Failed to load data. Please check file paths.")
        return
    
    # Data analysis
    analyze_data(expert_data, policy_data)
    
    # Create save paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    stiffness_plot_path = results_dir / "distance_vs_stiffness_comparison.png"
    damping_plot_path = results_dir / "distance_vs_damping_comparison.png"
    Kz_plot_path = results_dir / "distance_vs_Kz_comparison.png"

    plot_stiffness_comparison(expert_data, policy_data, stiffness_plot_path)
    
    plot_damping_comparison(expert_data, policy_data, damping_plot_path)

    plot_Kz_comparison(expert_data, policy_data, Kz_plot_path)
    
    print(f"\nPlots saved to: {results_dir}")

if __name__ == "__main__":
    main()