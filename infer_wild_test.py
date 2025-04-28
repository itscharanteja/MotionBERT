"""
This script is a comprehensive pipeline for analyzing human motion in videos using 2D-to-3D pose estimation. 
It processes video and AlphaPose JSON files, applies a deep learning model (MotionBERT) to reconstruct 3D poses, 
applies temporal smoothing, computes kinematic features, and generates detailed visualizations and summary statistics. 
It is designed for biomechanics or sports analysis, such as cycling motion
"""



import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_wild import WildDetDataset
from lib.utils.vismo import render_and_save
from multiprocessing import freeze_support
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

mp.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('--smooth_window', type=int, default=15, help='window size for temporal smoothing')
    parser.add_argument('--smooth_polyorder', type=int, default=3, help='polynomial order for Savitzky-Golay filter')
    opts = parser.parse_args()
    return opts

def temporal_smoothing(poses, window_size=15, polyorder=3):
    """
    Apply Savitzky-Golay filter for temporal smoothing of 3D poses
    
    Args:
        poses: numpy array of shape (T, J, 3) where T is time, J is joints, 3 is XYZ
        window_size: window size for the filter (must be odd number)
        polyorder: polynomial order for the filter
        
    Returns:
        smoothed poses of same shape as input
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Copy the poses to avoid modifying the original
    smoothed_poses = np.copy(poses)
    
    # For each joint and each dimension, apply the filter
    n_joints = poses.shape[1]
    for j in range(n_joints):
        for d in range(3):  # x, y, z dimensions
            smoothed_poses[:, j, d] = savgol_filter(
                poses[:, j, d], 
                window_size, 
                polyorder, 
                mode='interp'  # Use 'interp' to handle the edges
            )
    
    return smoothed_poses

def calculate_joint_angles(poses, parent_indices):
    """
    Calculate joint angles for the given poses
    
    Args:
        poses: numpy array of shape (T, J, 3) where T is time, J is joints, 3 is XYZ
        parent_indices: list defining the parent joint for each joint to form limbs
        
    Returns:
        joint_angles: angles between adjacent limbs in degrees (T, J-1)
    """
    n_frames, n_joints, _ = poses.shape
    joint_angles = np.zeros((n_frames, n_joints-1))
    
    # For each frame
    for f in range(n_frames):
        angle_idx = 0
        # For each joint (except root)
        for j in range(1, n_joints):
            # Skip if parent is -1 (no parent)
            if parent_indices[j] == -1:
                continue
                
            # Get the parent of the current joint
            parent = parent_indices[j]
            
            # Get the parent of the parent (grandparent)
            grandparent = parent_indices[parent]
            
            # Skip if grandparent is -1 (no grandparent)
            if grandparent == -1:
                continue
                
            # Calculate vectors
            v1 = poses[f, parent] - poses[f, grandparent]
            v2 = poses[f, j] - poses[f, parent]
            
            # Normalize the vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            # Avoid division by zero
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                angle = 0
            else:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Calculate the dot product
                dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                
                # Calculate the angle in degrees
                angle = np.degrees(np.arccos(dot_product))
            
            joint_angles[f, angle_idx] = angle
            angle_idx += 1
            
    return joint_angles

def calculate_angular_velocity(joint_angles, fps):
    """
    Calculate angular velocity from joint angles
    
    Args:
        joint_angles: numpy array of shape (T, J) where T is time, J is joints
        fps: frames per second of the video
        
    Returns:
        angular_velocity: angular velocity in degrees per second (T-1, J)
    """
    # Calculate time step
    dt = 1.0 / fps
    
    # Calculate the difference in angles between consecutive frames
    angle_diff = np.diff(joint_angles, axis=0)
    
    # Calculate angular velocity in degrees per second
    angular_velocity = angle_diff / dt
    
    return angular_velocity

def plot_kinematic_features(joint_angles, angular_velocity, out_path, joint_names=None):
    """
    Plot kinematic features and save to files
    
    Args:
        joint_angles: numpy array of shape (T, J)
        angular_velocity: numpy array of shape (T-1, J)
        out_path: path to save the plots
        joint_names: list of joint names (optional)
    """
    n_joints = joint_angles.shape[1]
    
    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(n_joints)]
    
    # Plot joint angles
    plt.figure(figsize=(12, 6))
    for j in range(n_joints):
        plt.plot(joint_angles[:, j], label=joint_names[j])
    
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angles Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'joint_angles.png'))
    plt.close()
    
    # Plot angular velocity
    plt.figure(figsize=(12, 6))
    for j in range(n_joints):
        plt.plot(angular_velocity[:, j], label=joint_names[j])
    
    plt.xlabel('Frame')
    plt.ylabel('Angular Velocity (degrees/s)')
    plt.title('Joint Angular Velocity Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'angular_velocity.png'))
    plt.close()

def calculate_mpjpe(pred_3d, gt_3d):
    """
    Calculate Mean Per Joint Position Error (MPJPE)
    
    Args:
        pred_3d: predicted 3D poses (T, J, 3)
        gt_3d: ground truth 3D poses (T, J, 3)
        
    Returns:
        overall_mpjpe: scalar, average MPJPE across all joints and frames (mm)
        per_joint_mpjpe: array of shape (J,), MPJPE for each joint (mm)
    """
    assert pred_3d.shape == gt_3d.shape, f"Shape mismatch: {pred_3d.shape} vs {gt_3d.shape}"
    
    # Calculate Euclidean distance for each joint in each frame
    error = np.sqrt(np.sum((pred_3d - gt_3d) ** 2, axis=2))  # (T, J)
    
    # Average error across frames for each joint
    per_joint_mpjpe = np.mean(error, axis=0)  # (J,)
    
    # Overall average error
    overall_mpjpe = np.mean(error)
    
    return overall_mpjpe, per_joint_mpjpe

def plot_mpjpe_comparison(unsmoothed_mpjpe, smoothed_mpjpe, joint_names, out_path):
    """
    Plot MPJPE comparison between smoothed and unsmoothed results
    
    Args:
        unsmoothed_mpjpe: per-joint MPJPE for unsmoothed results (J,)
        smoothed_mpjpe: per-joint MPJPE for smoothed results (J,)
        joint_names: list of joint names
        out_path: path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(joint_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, unsmoothed_mpjpe, width, label='Unsmoothed')
    bars2 = plt.bar(x + width/2, smoothed_mpjpe, width, label='Smoothed')
    
    plt.xlabel('Joints')
    plt.ylabel('MPJPE (mm)')
    plt.title('Per-Joint MPJPE Comparison: Smoothed vs. Unsmoothed')
    plt.xticks(x, joint_names, rotation=45, ha='right')
    plt.legend()
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_path, 'mpjpe_comparison.png'), dpi=300)
    plt.close()

def plot_kinematic_comparison(unsmoothed_angles, smoothed_angles, 
                              unsmoothed_velocity, smoothed_velocity,
                              joint_names, out_path, sample_frames=None):
    """
    Plot comparison of kinematic features between smoothed and unsmoothed results
    
    Args:
        unsmoothed_angles: joint angles for unsmoothed poses (T, J)
        smoothed_angles: joint angles for smoothed poses (T, J)
        unsmoothed_velocity: angular velocity for unsmoothed poses (T-1, J)
        smoothed_velocity: angular velocity for smoothed poses (T-1, J)
        joint_names: list of joint names
        out_path: path to save plots
        sample_frames: specific frames to annotate with angle values (optional)
    """
    n_joints = unsmoothed_angles.shape[1]
    
    # If sample_frames not specified, choose 5 evenly spaced frames
    if sample_frames is None:
        total_frames = len(unsmoothed_angles)
        sample_frames = [int(i * total_frames / 6) for i in range(1, 6)]
    
    # Create directory for individual joint plots if many joints
    if n_joints > 5:
        joint_plots_dir = os.path.join(out_path, 'joint_plots')
        os.makedirs(joint_plots_dir, exist_ok=True)
    
    # 1. Overall summary plot
    plt.figure(figsize=(15, 10))
    
    # Calculate average angle difference
    angle_diff = np.abs(smoothed_angles - unsmoothed_angles)
    avg_angle_diff = np.mean(angle_diff, axis=0)
    
    # Plot average angle difference
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(avg_angle_diff)), avg_angle_diff)
    plt.xlabel('Joint')
    plt.ylabel('Average Angle Difference (degrees)')
    plt.title('Average Difference in Joint Angles: Smoothed vs. Unsmoothed')
    plt.xticks(range(len(avg_angle_diff)), joint_names, rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}°', ha='center', va='bottom', fontsize=8)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calculate average velocity difference
    vel_diff = np.abs(smoothed_velocity - unsmoothed_velocity)
    avg_vel_diff = np.mean(vel_diff, axis=0)
    
    # Plot average velocity difference
    plt.subplot(2, 1, 2)
    bars = plt.bar(range(len(avg_vel_diff)), avg_vel_diff)
    plt.xlabel('Joint')
    plt.ylabel('Average Angular Velocity Difference (deg/s)')
    plt.title('Average Difference in Angular Velocity: Smoothed vs. Unsmoothed')
    plt.xticks(range(len(avg_vel_diff)), joint_names, rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'kinematic_diff_summary.png'), dpi=300)
    plt.close()
    
    # 2. Plot individual joint angles with detailed annotations
    for j in range(min(n_joints, 5)):  # Plot first 5 joints in main directory
        plt.figure(figsize=(15, 6))
        
        # Plot angles
        plt.plot(unsmoothed_angles[:, j], 'r-', label=f'Unsmoothed', alpha=0.7)
        plt.plot(smoothed_angles[:, j], 'b-', label=f'Smoothed', alpha=0.7)
        
        # Add annotations for selected frames
        for frame in sample_frames:
            if frame < len(unsmoothed_angles):
                # Unsmoothed angle annotation
                plt.plot(frame, unsmoothed_angles[frame, j], 'ro', markersize=8)
                plt.text(frame, unsmoothed_angles[frame, j] + 2, 
                        f'{unsmoothed_angles[frame, j]:.1f}°', 
                        color='r', fontsize=9, ha='center')
                
                # Smoothed angle annotation
                plt.plot(frame, smoothed_angles[frame, j], 'bo', markersize=8)
                plt.text(frame, smoothed_angles[frame, j] - 4, 
                        f'{smoothed_angles[frame, j]:.1f}°', 
                        color='b', fontsize=9, ha='center')
                
                # Draw line to highlight difference
                plt.plot([frame, frame], 
                        [unsmoothed_angles[frame, j], smoothed_angles[frame, j]], 
                        'k--', alpha=0.5)
                
                # Calculate and show difference
                diff = abs(smoothed_angles[frame, j] - unsmoothed_angles[frame, j])
                mid_y = (smoothed_angles[frame, j] + unsmoothed_angles[frame, j]) / 2
                plt.text(frame + 5, mid_y, f'Δ{diff:.1f}°', 
                        color='green', fontsize=9, ha='left', va='center')
        
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title(f'Joint Angle Comparison: {joint_names[j]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add overall statistics in a text box
        mean_diff = np.mean(np.abs(smoothed_angles[:, j] - unsmoothed_angles[:, j]))
        max_diff = np.max(np.abs(smoothed_angles[:, j] - unsmoothed_angles[:, j]))
        stats_text = f'Mean Difference: {mean_diff:.2f}°\nMax Difference: {max_diff:.2f}°'
        
        plt.figtext(0.15, 0.15, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f'angle_comparison_{joint_names[j]}.png'), dpi=300)
        plt.close()
    
    # Plot remaining joints in subdirectory if too many
    if n_joints > 5:
        for j in range(5, n_joints):
            plt.figure(figsize=(15, 6))
            plt.plot(unsmoothed_angles[:, j], 'r-', label=f'Unsmoothed', alpha=0.7)
            plt.plot(smoothed_angles[:, j], 'b-', label=f'Smoothed', alpha=0.7)
            
            # Add annotations for selected frames (simplified for additional joints)
            for frame in sample_frames:
                if frame < len(unsmoothed_angles):
                    plt.plot(frame, unsmoothed_angles[frame, j], 'ro', markersize=6)
                    plt.plot(frame, smoothed_angles[frame, j], 'bo', markersize=6)
            
            plt.xlabel('Frame')
            plt.ylabel('Angle (degrees)')
            plt.title(f'Joint Angle Comparison: {joint_names[j]}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(joint_plots_dir, f'angle_comparison_{joint_names[j]}.png'), dpi=300)
            plt.close()
    
    # 3. Create heatmap to visualize angle differences over time
    plt.figure(figsize=(15, 8))
    im = plt.imshow(angle_diff.T, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Angle Difference (degrees)')
    plt.xlabel('Frame')
    plt.ylabel('Joint')
    plt.title('Joint Angle Differences Over Time (Smoothed - Unsmoothed)')
    plt.yticks(range(n_joints), joint_names)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'angle_diff_heatmap.png'), dpi=300)
    plt.close()

opts = parse_args()
args = get_config(opts.config)

model_backbone = load_backbone(args)
if torch.cuda.is_available():
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()

print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)

# Remove 'module.' prefix if it exists
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model_pos'].items():
    name = k.replace('module.', '')  # remove 'module.' prefix
    new_state_dict[name] = v

model_backbone.load_state_dict(new_state_dict, strict=True)
model_pos = model_backbone
model_pos.eval()
testloader_params = {
          'batch_size': 1,
          'shuffle': False,
          'num_workers': 0,
          'pin_memory': True,
          'drop_last': False
}

vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
fps_in = vid.get_meta_data()['fps']
vid_size = vid.get_meta_data()['size']
os.makedirs(opts.out_path, exist_ok=True)

if opts.pixel:
    # Keep relative scale with pixel coornidates
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
else:
    # Scale to [-1,1]
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

test_loader = DataLoader(wild_dataset, **testloader_params)

if __name__ ==  "__main__":
    freeze_support()

    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
            else:
                predicted_3d_pos[:,0,0,2]=0
                pass
            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    
    # Save original unsmoothed results
    unsmoothed_results = np.copy(results_all)
    render_and_save(unsmoothed_results, '%s/X3D_unsmoothed.mp4' % (opts.out_path), keep_imgs=False, fps=fps_in)
    
    # Apply temporal smoothing
    print("Applying temporal smoothing...")
    smoothed_results = temporal_smoothing(
        results_all, 
        window_size=opts.smooth_window, 
        polyorder=opts.smooth_polyorder
    )
    
    # Save smoothed results
    render_and_save(smoothed_results, '%s/X3D_smoothed.mp4' % (opts.out_path), keep_imgs=False, fps=fps_in)
    
    # Convert to pixel coordinates if needed
    if opts.pixel:
        # Convert to pixel coordinates
        unsmoothed_results = unsmoothed_results * (min(vid_size) / 2.0)
        unsmoothed_results[:,:,:2] = unsmoothed_results[:,:,:2] + np.array(vid_size) / 2.0
        
        smoothed_results = smoothed_results * (min(vid_size) / 2.0)
        smoothed_results[:,:,:2] = smoothed_results[:,:,:2] + np.array(vid_size) / 2.0
    
    # Save both versions as .npy files
    np.save('%s/X3D_unsmoothed.npy' % (opts.out_path), unsmoothed_results)
    np.save('%s/X3D_smoothed.npy' % (opts.out_path), smoothed_results)
    
    # Define parent indices for H36M skeleton (adjust as needed for your skeleton)
    # The array specifies the parent joint index for each joint
    # Example: [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    # -1 means no parent (root joint)
    parent_indices = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    
    # Joint names for better visualization (adjust as needed)
    joint_names = [
        "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", 
        "Spine", "Thorax", "Neck", "Head", "LShoulder", "LElbow", 
        "LWrist", "RShoulder", "RElbow", "RWrist"
    ]
    
    # Calculate kinematic features from smoothed poses
    print("Calculating kinematic features for smoothed results...")
    joint_angles = calculate_joint_angles(smoothed_results, parent_indices)
    angular_velocity = calculate_angular_velocity(joint_angles, fps_in)
    
    # Save smoothed kinematic features
    np.save('%s/joint_angles.npy' % (opts.out_path), joint_angles)
    np.save('%s/angular_velocity.npy' % (opts.out_path), angular_velocity)
    
    # Calculate kinematic features for unsmoothed as well
    print("Calculating kinematic features for unsmoothed results...")
    unsmoothed_joint_angles = calculate_joint_angles(unsmoothed_results, parent_indices)
    unsmoothed_angular_velocity = calculate_angular_velocity(unsmoothed_joint_angles, fps_in)
    
    # Save unsmoothed kinematic features
    np.save('%s/unsmoothed_joint_angles.npy' % (opts.out_path), unsmoothed_joint_angles)
    np.save('%s/unsmoothed_angular_velocity.npy' % (opts.out_path), unsmoothed_angular_velocity)
    
    # Plot the individual kinematic features
    joint_pair_names = [f"{a}-{b}" for a, b in zip(joint_names[1:], joint_names)]
    plot_kinematic_features(joint_angles, angular_velocity, opts.out_path, joint_pair_names)
    
    # Create comparison plots between smoothed and unsmoothed kinematic features
    print("Creating comparison plots...")
    plot_kinematic_comparison(
        unsmoothed_joint_angles, 
        joint_angles,
        unsmoothed_angular_velocity, 
        angular_velocity,
        joint_pair_names, 
        opts.out_path,
        sample_frames=[30, 60, 90, 120, 150]  # Sample frames for detailed analysis
    )
    
    # Calculate MPJPE between smoothed and unsmoothed as a measure of the smoothing effect
    # Using unsmoothed as "ground truth" to see how much the smoothing changed the poses
    print("Calculating MPJPE between smoothed and unsmoothed results...")
    overall_mpjpe, per_joint_mpjpe = calculate_mpjpe(smoothed_results, unsmoothed_results)
    
    print(f"Overall MPJPE between smoothed and unsmoothed: {overall_mpjpe:.2f} units")
    plot_mpjpe_comparison(
        np.zeros_like(per_joint_mpjpe),  # zeros for unsmoothed (as it's our reference)
        per_joint_mpjpe,                 # show how much smoothing affected each joint
        joint_names,
        opts.out_path
    )
    
    # Add additional summary statistics
    mean_angle_diff = np.mean(np.abs(joint_angles - unsmoothed_joint_angles))
    max_angle_diff = np.max(np.abs(joint_angles - unsmoothed_joint_angles))
    mean_velocity_diff = np.mean(np.abs(angular_velocity - unsmoothed_angular_velocity))
    max_velocity_diff = np.max(np.abs(angular_velocity - unsmoothed_angular_velocity))
    
    # Create a summary file with key metrics
    with open(os.path.join(opts.out_path, 'smoothing_analysis_summary.txt'), 'w') as f:
        f.write("## SMOOTHING ANALYSIS SUMMARY ##\n\n")
        f.write(f"Smoothing window size: {opts.smooth_window}\n")
        f.write(f"Smoothing polynomial order: {opts.smooth_polyorder}\n\n")
        
        f.write("## POSE CHANGES ##\n")
        f.write(f"Overall MPJPE between smoothed and unsmoothed: {overall_mpjpe:.4f} units\n")
        f.write(f"Max joint displacement from smoothing: {np.max(per_joint_mpjpe):.4f} units\n\n")
        
        f.write("## KINEMATIC CHANGES ##\n")
        f.write(f"Mean joint angle difference: {mean_angle_diff:.4f} degrees\n")
        f.write(f"Max joint angle difference: {max_angle_diff:.4f} degrees\n")
        f.write(f"Mean angular velocity difference: {mean_velocity_diff:.4f} degrees/s\n")
        f.write(f"Max angular velocity difference: {max_velocity_diff:.4f} degrees/s\n")
    
    print(f"Processing complete. Results saved to {opts.out_path}")
    print(f"- Unsmoothed video: X3D_unsmoothed.mp4")
    print(f"- Smoothed video: X3D_smoothed.mp4") 
    print(f"- Kinematic features: joint_angles.png, angular_velocity.png")
    print(f"- Comparison plots: angle_comparison_*.png, kinematic_diff_summary.png, mpjpe_comparison.png")
    print(f"- Summary report: smoothing_analysis_summary.txt")