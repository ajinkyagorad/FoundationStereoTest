
# license retained from original demo code

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# $ python scripts/run_vid.py  --ckpt_dir ./pretrained_models/model_best_bp2.pth  --intrinsic_file scripts/camera_calibration.json 



import os
import sys
import argparse
import cv2
import numpy as np
import torch
import json
import open3d as o3d
import time
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

# Undistort image function
def undistort_image(img, K, dist):
    """Undistorts an image using the camera intrinsic matrix and distortion coefficients."""
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), None)
    undistorted = cv2.undistort(img, K, dist, None, new_camera_mtx)
    return undistorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsic_file', default='./assets/K.json', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default='./pretrained_models/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--z_far', type=float, default=10.0, help='maximum depth range for valid points')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    ckpt_dir = args.ckpt_dir
    args = OmegaConf.create(vars(args))
    logging.info(f"Using pretrained model from {ckpt_dir}")

    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    model = FoundationStereo(cfg)

    ckpt = torch.load(ckpt_dir)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    # Open camera streams
    cap_left = cv2.VideoCapture(0)  # Left camera
    cap_right = cv2.VideoCapture(4)  # Right camera

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open cameras.")
        exit()

    # Load calibration data
    with open(args.intrinsic_file, 'r') as f:
        calib_data = json.load(f)
        K = np.array(calib_data['camera_matrix'], dtype=np.float32)
        dist = np.array(calib_data['dist_coeff'], dtype=np.float32)
        baseline = float(calib_data.get('baseline', 1.0))

    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create an empty point cloud object
    pcd = o3d.geometry.PointCloud()

    # Timing setup for continuous updates
    start_time = time.time()
    prev_sec = -1
    translation_index = 0
    translation_speed = 0.01
    first_call = True

    while True:
        curr_sec = int(time.time() - start_time)
        if curr_sec - prev_sec >= 1.0:  # Check if 1 second has passed
            prev_sec = curr_sec

            ret_left, img_left = cap_left.read()
            ret_right, img_right = cap_right.read()
            if not ret_left or not ret_right:
                print("Failed to capture frames")
                break

            # Undistort images
            img_left = undistort_image(img_left, K, dist)
            img_right = undistort_image(img_right, K, dist)

            # Resize images if scale is provided
            if args.scale != 1:
                img_left = cv2.resize(img_left, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
                img_right = cv2.resize(img_right, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)

            # Convert images to tensors
            img_left_tensor = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
            img_right_tensor = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)

            # Pad images to be divisible by 32
            padder = InputPadder(img_left_tensor.shape, divis_by=32, force_square=False)
            img_left_padded, img_right_padded = padder.pad(img_left_tensor, img_right_tensor)

            # Forward pass through the model
            with torch.cuda.amp.autocast(True):
                disp = model.forward(img_left_padded, img_right_padded, iters=args.valid_iters, test_mode=True)

            # Unpad disparity map
            disp = padder.unpad(disp.float()).squeeze().cpu().numpy()

            # Calculate depth map
            depth = K[0, 0] * baseline / (disp + 1e-6)

            # Convert depth map to 3D point cloud
            xyz_map = depth2xyzmap(depth, K)
            valid_mask = ~np.isnan(xyz_map[:, :, 2]) & (xyz_map[:, :, 2] > 0) & (xyz_map[:, :, 2] < args.z_far)

            # Resize left image to match xyz_map dimensions
            img_left_resized = cv2.resize(img_left, (xyz_map.shape[1], xyz_map.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Filter valid points and corresponding colors
            xyz_map_filtered = xyz_map[valid_mask]
            colors_filtered = img_left_resized.reshape(-1, 3)[valid_mask.reshape(-1)] / 255.0

            if xyz_map_filtered.shape[0] == 0:
                print("No valid 3D points found! Check camera setup and disparity output.")
                continue

            # Update point cloud with new 3D points
            pcd.points = o3d.utility.Vector3dVector(xyz_map_filtered)
            pcd.colors = o3d.utility.Vector3dVector(colors_filtered)

            # Add or update geometry objects
            if first_call:
                vis.add_geometry(pcd)  # Add geometry once in the first loop
                first_call = False
            else:
                vis.update_geometry(pcd)  # Update geometry continuously

            translation_index += 1  # Optional, if you want to simulate movement

        # Update visualizer window continuously
        vis.poll_events()
        vis.update_renderer()

    # Release resources
    cap_left.release()
    cap_right.release()
    vis.destroy_window()
    cv2.destroyAllWindows()
