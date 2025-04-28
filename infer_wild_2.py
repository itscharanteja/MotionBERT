from collections import OrderedDict
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
    parser.add_argument(
        "--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin',
                        type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str,
                        help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true',
                        help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None,
                        help='target person id')
    parser.add_argument('--clip_len', type=int, default=243,
                        help='clip length for network input')
    parser.add_argument('--smooth_window', type=int, default=15,
                        help='window size for temporal smoothing')
    parser.add_argument('--smooth_polyorder', type=int, default=3,
                        help='polynomial order for Savitzky-Golay filter')
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


opts = parse_args()
args = get_config(opts.config)

model_backbone = load_backbone(args)
if torch.cuda.is_available():
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()

print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(
    opts.evaluate, map_location=lambda storage, loc: storage)

# Remove 'module.' prefix if it exists
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

# Extract video filename without extension
video_basename = os.path.splitext(os.path.basename(opts.vid_path))[0]

if opts.pixel:
    # Keep relative scale with pixel coornidates
    wild_dataset = WildDetDataset(
        opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
else:
    # Scale to [-1,1]
    wild_dataset = WildDetDataset(
        opts.json_path, clip_len=opts.clip_len, scale_range=[1, 1], focus=opts.focus)

test_loader = DataLoader(wild_dataset, **testloader_params)

if __name__ == "__main__":
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
                predicted_3d_pos_2 = flip_data(
                    predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 +
                                    predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                # [N,T,17,3]
                predicted_3d_pos[:, :, 0, :] = 0
            else:
                predicted_3d_pos[:, 0, 0, 2] = 0
                pass
            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)

    # Save original unsmoothed results
    unsmoothed_results = np.copy(results_all)
    unsmoothed_video_path = f'{opts.out_path}/X3D_US_{video_basename}.mp4'
    render_and_save(unsmoothed_results, unsmoothed_video_path,
                    keep_imgs=False, fps=fps_in)

    # Apply temporal smoothing
    print("Applying temporal smoothing...")
    smoothed_results = temporal_smoothing(
        results_all,
        window_size=opts.smooth_window,
        polyorder=opts.smooth_polyorder
    )

    # Save smoothed results
    smoothed_video_path = f'{opts.out_path}/X3D_S_{video_basename}.mp4'
    render_and_save(smoothed_results, smoothed_video_path,
                    keep_imgs=False, fps=fps_in)

    # Convert to pixel coordinates if needed
    if opts.pixel:
        # Convert to pixel coordinates
        unsmoothed_results = unsmoothed_results * (min(vid_size) / 2.0)
        unsmoothed_results[:, :, :2] = unsmoothed_results[:,
                                                          :, :2] + np.array(vid_size) / 2.0

        smoothed_results = smoothed_results * (min(vid_size) / 2.0)
        smoothed_results[:, :, :2] = smoothed_results[:,
                                                      :, :2] + np.array(vid_size) / 2.0

    # Save both versions as .npy files
    np.save(f'{opts.out_path}/X3D_US_{video_basename}.npy', unsmoothed_results)
    np.save(f'{opts.out_path}/X3D_S_{video_basename}.npy', smoothed_results)

    parent_indices = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

    # Joint names for better visualization (adjust as needed)
    joint_names = [
        "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
        "Spine", "Thorax", "Neck", "Head", "LShoulder", "LElbow",
        "LWrist", "RShoulder", "RElbow", "RWrist"
    ]
    print(f"Processing complete. Results saved to {opts.out_path}")
    print(f"- Unsmoothed video: X3D_US_{video_basename}.mp4")
    print(f"- Smoothed video: X3D_S_{video_basename}.mp4")
