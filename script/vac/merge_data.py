#!/usr/bin/env python3
"""
Variable Admittance Control Expert Data Merger

This script merges multiple expert trajectory PKL files into a single demonstration dataset.
The merged dataset combines observations and actions from all trajectories while preserving
the Variable Admittance Control format for AIRL training.

Usage:
    python merge_data.py                           # Merge all expert_pih_*.pkl files 
    python merge_data.py --task pih                # Merge all expert_pih_*.pkl files
    python merge_data.py --pattern expert_pih_     # Merge files matching pattern
    python merge_data.py --files 0,1,2,3,4        # Merge specific numbered files
    python merge_data.py --output custom_demo.pkl # Custom output filename
"""

import os
import sys
import glob
import pickle
import numpy as np
import argparse
from typing import List, Dict, Any

def find_expert_files(data_dir: str, task: str = 'pih', pattern: str = None, file_indices: List[int] = None) -> List[str]:
    """
    Find expert trajectory files based on criteria
    
    Args:
        data_dir: Directory containing expert files
        task: Task name (e.g., 'pih', 'regulation')  
        pattern: Custom file pattern (e.g., 'expert_pih_')
        file_indices: Specific file indices to merge
        
    Returns:
        List of expert file paths
    """
    if file_indices is not None:
        # Merge specific numbered files
        expert_files = []
        for idx in file_indices:
            filename = f'expert_{task}_{idx}.pkl'
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                expert_files.append(filepath)
            else:
                print(f"Warning: File {filename} not found, skipping...")
        return expert_files
    
    elif pattern is not None:
        # Use custom pattern
        search_pattern = os.path.join(data_dir, f'{pattern}*.pkl')
    else:
        # Default: all expert files for specified task
        search_pattern = os.path.join(data_dir, f'expert_{task}_*.pkl')
    
    expert_files = glob.glob(search_pattern)
    expert_files.sort()  # Sort for consistent ordering
    
    return expert_files

def load_expert_data(filepath: str) -> Dict[str, Any]:
    """Load and validate expert trajectory data"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Validate required keys
        required_keys = ['observations', 'actions', 'metadata']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate data shapes
        obs = np.array(data['observations'])
        acts = np.array(data['actions'])
        
        if obs.shape[0] != acts.shape[0]:
            raise ValueError(f"Observation and action lengths don't match: {obs.shape[0]} vs {acts.shape[0]}")
        
        # Expected dimensions for Variable Admittance Control
        if obs.shape[1] != 6:
            print(f"Warning: Expected 6D observations, got {obs.shape[1]}D in {os.path.basename(filepath)}")
        
        if acts.shape[1] != 7:
            print(f"Warning: Expected 7D actions, got {acts.shape[1]}D in {os.path.basename(filepath)}")
        
        print(f"✓ Loaded {os.path.basename(filepath)}: {obs.shape[0]} samples, obs={obs.shape[1]}D, act={acts.shape[1]}D")
        return data
        
    except Exception as e:
        print(f"✗ Error loading {filepath}: {e}")
        return None

def merge_expert_data(expert_files: List[str]) -> Dict[str, Any]:
    """
    Merge multiple expert trajectory datasets
    
    Args:
        expert_files: List of expert PKL file paths
        
    Returns:
        Merged dataset in Variable Admittance Control format
    """
    if not expert_files:
        raise ValueError("No expert files to merge")
    
    print(f"\n=== Merging {len(expert_files)} expert trajectory files ===")
    
    all_observations = []
    all_actions = []
    all_metadata = []
    total_samples = 0
    
    # Load and combine data from all files
    for filepath in expert_files:
        data = load_expert_data(filepath)
        if data is None:
            continue
            
        # Convert to numpy arrays for validation
        obs = np.array(data['observations'])
        acts = np.array(data['actions'])
        
        # Accumulate data
        all_observations.extend(data['observations'])
        all_actions.extend(data['actions'])
        all_metadata.append(data['metadata'])
        total_samples += obs.shape[0]
        
        print(f"  Added: {obs.shape[0]} samples from {os.path.basename(filepath)}")
    
    if not all_observations:
        raise ValueError("No valid data found in expert files")
    
    # Create merged dataset
    merged_data = {
        'observations': all_observations,
        'actions': all_actions,
        'metadata': create_merged_metadata(all_metadata, total_samples, expert_files)
    }
    
    # Validate merged data
    validate_merged_data(merged_data)
    
    print(f"\n✓ Successfully merged {len(expert_files)} files")
    print(f"✓ Total samples: {total_samples}")
    
    return merged_data

def create_merged_metadata(metadata_list: List[Dict], total_samples: int, expert_files: List[str] = None) -> Dict[str, Any]:
    """Create metadata for merged dataset"""
    if not metadata_list:
        raise ValueError("No metadata to merge")
    
    # Use first metadata as template
    base_metadata = metadata_list[0].copy()
    
    # Extract trajectory information
    trajectory_indices = []
    tasks = set()
    
    # Extract trajectory indices from filenames instead of metadata for reliability
    if expert_files:
        for filepath in expert_files:
            filename = os.path.basename(filepath)
            # Extract index from filename like "expert_pih_25.pkl"
            try:
                if filename.startswith('expert_') and filename.endswith('.pkl'):
                    parts = filename[:-4].split('_')  # Remove .pkl and split
                    if len(parts) >= 3:
                        trajectory_indices.append(int(parts[-1]))  # Last part should be index
            except (ValueError, IndexError):
                print(f"Warning: Could not extract index from {filename}")
    else:
        # Fallback to metadata (original behavior)
        for meta in metadata_list:
            if 'trajectory_index' in meta:
                trajectory_indices.append(meta['trajectory_index'])
    
    for meta in metadata_list:
        if 'task' in meta:
            tasks.add(meta['task'])
    
    # Update merged metadata
    merged_metadata = {
        'observation_dim': base_metadata.get('observation_dim', 6),
        'action_dim': base_metadata.get('action_dim', 7), 
        'trajectory_length': total_samples,
        'num_trajectories': len(metadata_list),
        'trajectory_indices': trajectory_indices,
        'tasks': list(tasks),
        'observation_description': base_metadata.get('observation_description', 
            'Variable Admittance Control: [e_t(3), pd_t(3)] where e_t=p-pd, pd_t=desired_pos'),
        'action_description': base_metadata.get('action_description',
            'impedance_parameters: [K1, K2, K3, K4, K5, K6, damping_ratio]'),
        'error_convention': base_metadata.get('error_convention',
            'tracking_error: actual - desired (positive means overshoot)'),
        'control_law': base_metadata.get('control_law',
            'Variable Admittance: pd_new = pd + e_admittance'),
        'merge_timestamp': __import__('datetime').datetime.now().isoformat()
    }
    
    return merged_metadata

def validate_merged_data(merged_data: Dict[str, Any]):
    """Validate merged dataset"""
    obs = np.array(merged_data['observations'])
    acts = np.array(merged_data['actions'])
    
    print(f"\n=== Dataset Validation ===")
    print(f"Observations shape: {obs.shape}")
    print(f"Actions shape: {acts.shape}")
    
    # Check data consistency
    if obs.shape[0] != acts.shape[0]:
        raise ValueError(f"Mismatched lengths: obs={obs.shape[0]}, acts={acts.shape[0]}")
    
    # Check for NaN or infinite values
    if np.isnan(obs).any():
        print("Warning: NaN values found in observations")
    if np.isnan(acts).any():
        print("Warning: NaN values found in actions")
    if np.isinf(obs).any():
        print("Warning: Infinite values found in observations") 
    if np.isinf(acts).any():
        print("Warning: Infinite values found in actions")
    
    # Variable Admittance Control statistics
    print(f"\n=== Variable Admittance Control Statistics ===")
    print(f"Observation statistics:")
    print(f"  Tracking error e_t range: [{obs[:, :3].min():.4f}, {obs[:, :3].max():.4f}]")
    print(f"  Desired position pd_t range: [{obs[:, 3:6].min():.4f}, {obs[:, 3:6].max():.4f}]")
    
    print(f"Action statistics:")
    print(f"  K1-K3 (position stiffness) range: [{acts[:, :3].min():.1f}, {acts[:, :3].max():.1f}]")
    print(f"  K4-K6 (rotation stiffness) range: [{acts[:, 3:6].min():.1f}, {acts[:, 3:6].max():.1f}]")
    print(f"  Damping ratio range: [{acts[:, 6].min():.3f}, {acts[:, 6].max():.3f}]")
    
    print("✓ Validation completed")

def save_merged_data(merged_data: Dict[str, Any], output_path: str):
    """Save merged dataset to PKL file"""
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(merged_data, f)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"\n✓ Merged dataset saved to: {output_path}")
        print(f"✓ File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ Error saving merged data: {e}")
        raise

def main():
    """Main function for merging expert trajectory data"""
    parser = argparse.ArgumentParser(description='Merge Variable Admittance Control Expert Trajectory Data')
    parser.add_argument('--task', default='pih', help='Task name (default: pih)')
    parser.add_argument('--pattern', help='Custom file pattern (e.g., expert_pih_)')
    parser.add_argument('--files', help='Comma-separated list of file indices (e.g., 0,1,2,3)')
    parser.add_argument('--output', default='expert_demonstration.pkl', help='Output filename (default: expert_demonstration.pkl)')
    parser.add_argument('--data-dir', help='Custom data directory path')
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Default: assume script is in script/vac/ and data is in ../../data/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', '..', 'data')
        data_dir = os.path.abspath(data_dir)
    
    if not os.path.exists(data_dir):
        print(f"✗ Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Data directory: {data_dir}")
    
    # Parse file indices if provided
    file_indices = None
    if args.files:
        try:
            file_indices = [int(x.strip()) for x in args.files.split(',')]
            print(f"Merging specific files: {file_indices}")
        except ValueError:
            print("✗ Error: Invalid file indices format. Use comma-separated integers (e.g., 0,1,2)")
            sys.exit(1)
    
    # Find expert files
    try:
        expert_files = find_expert_files(data_dir, args.task, args.pattern, file_indices)
        
        if not expert_files:
            print(f"✗ No expert files found matching criteria")
            if file_indices:
                print(f"  Looked for: expert_{args.task}_{{index}}.pkl where index in {file_indices}")
            elif args.pattern:
                print(f"  Looked for: {args.pattern}*.pkl")
            else:
                print(f"  Looked for: expert_{args.task}_*.pkl")
            sys.exit(1)
        
        print(f"Found {len(expert_files)} expert files:")
        for f in expert_files:
            print(f"  {os.path.basename(f)}")
        
    except Exception as e:
        print(f"✗ Error finding expert files: {e}")
        sys.exit(1)
    
    # Merge data
    try:
        merged_data = merge_expert_data(expert_files)
    except Exception as e:
        print(f"✗ Error merging data: {e}")
        sys.exit(1)
    
    # Save merged data
    output_path = os.path.join(data_dir, args.output)
    try:
        save_merged_data(merged_data, output_path)
    except Exception as e:
        print(f"✗ Error saving data: {e}")
        sys.exit(1)
    
    print(f"\n=== Merge Complete ===")
    print(f"✓ Combined {len(expert_files)} trajectories")
    print(f"✓ Total samples: {len(merged_data['observations'])}")
    print(f"✓ Output file: {args.output}")
    print(f"✓ Location: {data_dir}")

if __name__ == '__main__':
    main()