import sys
import os
import glob
import pathlib
from sympy import O
import torch

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import random
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R


def toDisplay(x, target_dim = 2):
    while(x.dim() > target_dim):
        x = x[0]
    return x.detach().cpu().numpy()


def plot_multi_np(plist):
    """
    Args: plist, list of numpy arrays of shape, (1,num_points,3)
    """
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#e377c2',  # raspberry yogurt pink
        '#8c564b',  # chestnut brown
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'   # blue-teal
    ]
    skip = 1
    go_data = []
    for i in range(len(plist)):
        p_dp = toDisplay(torch.from_numpy(plist[i]))
        plot = go.Scatter3d(x=p_dp[::skip,0], y=p_dp[::skip,1], z=p_dp[::skip,2], 
                     mode='markers', marker=dict(size=1, color=colors[i],
                     symbol='circle'))
        go_data.append(plot)
 
    layout = go.Layout(
        scene=dict(
            aspectmode='data'
        )
    )

    fig = go.Figure(data=go_data, layout=layout)
    fig.show()
    return fig


def load_shelf_point_cloud(filepath):
    """
    Load shelf point cloud and remaining points from given file path.
    
    Parameters:
    - filepath: file path to .npz file containing point cloud data.
    
    Returns:
    - shelf point cloud.
    - remaining point cloud.
    """
    data = np.load(filepath)
    points = data['clouds']
    classes = data['classes']
    
    # Get the shelf points
    shelf_points = points[classes == 1]
    other_points = points[classes != 1]
    
    return shelf_points, other_points
    
    
def solve_for_plane(data):
    """Solves the system Pc = q for plane coefficients c (a, b, c, d)."""
    
    A = np.zeros((data.shape[0], 3))
    for i in range(A.shape[0]):
        A[i, 0] = data[i, 0]
        A[i, 1] = data[i, 1]
        A[i, 2] = 1
        
    b = data[:, 2]
    
    # Form P and q
    P = A.T @ A
    q = A.T @ b
    
    # Solve for c
    U, E, V = np.linalg.svd(P)
    c = V.T @ np.diag(1 / E) @ U.T @ q
    
    # We assume coefficient c is -1
    final_c = np.zeros(4)
    final_c[0] = c[0]
    final_c[1] = c[1]
    final_c[2] = -1
    final_c[3] = c[2]
    
    return final_c
    
    
def do_plane_ransac(data, num_iters, num_samples, threshold):
    """Does RANSAC on the given data."""
    
    best_plane = None
    best_inliers = -1
    inlier_idxs = None
    
    # Do RANSAC
    for i in range(num_iters):
        # Sample num_samples points
        sample_indices = np.random.choice(data.shape[0], num_samples, replace=False)
        sample_data = data[sample_indices, :]
        
        coeffs = solve_for_plane(sample_data)
        
        a, b, c, d = coeffs
        distances = np.abs(a * data[:, 0] + b * data[:, 1] + c * data[:, 2] + d)
        distances /= np.sqrt(a**2 + b**2 + c**2)
        
        # Get the inliers
        inlier_indices = np.where(distances < threshold)[0]
        num_inliers = len(inlier_indices)
        
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_plane = coeffs
            inlier_idxs = inlier_indices
            
    inliner_points = data[inlier_idxs, :]
            
    return best_plane, inlier_idxs, inliner_points


def sample_points_within_polygon(polygon, num_points, sampling_method='random'):
    """
    Generate points inside the given polygon.
    
    Parameters:
    - polygon: A shapely Polygon object.
    - num_points: Number of points to generate.
    - sampling_method: Either 'random' or 'uniform'.
    
    Returns:
    - List of points inside the polygon.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    
    if sampling_method == 'random':
        points = []
        while len(points) < num_points:
            # Randomly sample a point within the bounding rectangle
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            
            if random_point.within(polygon):
                points.append([random_point.x, random_point.y])
    
    elif sampling_method == 'uniform':
        # Calculate the step size for the grid based on the desired number of points
        step_x = (max_x - min_x) / np.sqrt(num_points)
        step_y = (max_y - min_y) / np.sqrt(num_points)
        
        x_coords = np.arange(min_x, max_x, step_x)
        y_coords = np.arange(min_y, max_y, step_y)
        
        points = []
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                if point.within(polygon):
                    points.append([x, y])
                    
        # If we have more points than desired due to grid overlap, we'll trim the list
        if len(points) > num_points:
            points = points[:num_points]
    
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}. Choose either 'random' or 'uniform'.")
    
    return np.array(points)


def save_filled_shelf_point_cloud(filepath, savedir, augmented_pcd):
    """
    Save augmented shelf point cloud to given file path.
    
    Parameters:
    - filepath: file path to .npz file containing point cloud data.
    - savedir: directory to save the augmented point cloud.
    - augmented_pcd: augmented point cloud.
    """
    # Load the original point cloud data
    data = np.load(filepath, allow_pickle=True)
    clouds = data['clouds']
    classes = data['classes']
    colors = data['colors']
    
    # Remove the original mug points and corresponding classes and colors
    shelf_idxs = np.where(classes == 1)[0]
    clouds = np.delete(clouds, shelf_idxs, axis=0)
    classes = np.delete(classes, shelf_idxs, axis=0)
    colors = np.delete(colors, shelf_idxs, axis=0)
    
    # Add the augmented mug points and corresponding classes and colors
    clouds = np.vstack([clouds, augmented_pcd])
    classes = np.hstack([classes, np.full(augmented_pcd.shape[0], 1)])
    colors = np.vstack([colors, np.full((augmented_pcd.shape[0], 3), [0, 0, 255])])
    
    # Update the original data
    new_data = {'clouds': clouds, 'classes': classes, 'colors': colors}
    for key in data.keys():
        if key not in new_data:
            new_data[key] = data[key]
    
    # Save the augmented point cloud
    filename = os.path.basename(filepath)
    savepath = os.path.join(savedir, filename)
    os.makedirs(savedir, exist_ok=True)
    np.savez(savepath, **new_data)


def fill_shelf_surface(shelf_points, ransac_threshold=0.0005, num_sampled_points=10000):
    """
    Fills in holes in the shelf surface.

    Parameters:
    - shelf_points: shelf point cloud.
    - ransac_threshold: RANSAC threshold for plane fitting.
    - num_sampled_points: number of points to sample for the filled surface.
    
    Returns:
    - filled shelf point cloud.
    """
    
    # Get the two surfaces
    plane_1, plane_1_inlier_idxs, plane_1_inlier_points = do_plane_ransac(shelf_points, 100, 3, ransac_threshold)
    plane_1_remaining_points = np.delete(shelf_points, plane_1_inlier_idxs, axis=0)
    
    plane_2, plane_2_inlier_idxs, plane_2_inlier_points = do_plane_ransac(plane_1_remaining_points, 100, 3, ransac_threshold)
    plane_2_remaining_points = np.delete(plane_1_remaining_points, plane_2_inlier_idxs, axis=0)
    
    # Get which plane is the horizontal one
    plane_1_center = np.mean(plane_1_inlier_points, axis=0)
    plane_2_center = np.mean(plane_2_inlier_points, axis=0)
    
    if plane_1_center[2] > plane_2_center[2]:
        horizontal_plane = plane_1
        horizontal_plane_inlier_idxs = plane_1_inlier_idxs
        horizontal_plane_inlier_points = plane_1_inlier_points
        
        vertical_plane = plane_2
        vertical_plane_inlier_idxs = plane_2_inlier_idxs
        vertical_plane_inlier_points = plane_2_inlier_points
        
    else:
        horizontal_plane = plane_2
        horizontal_plane_inlier_idxs = plane_2_inlier_idxs
        horizontal_plane_inlier_points = plane_2_inlier_points
    
        vertical_plane = plane_1
        vertical_plane_inlier_idxs = plane_1_inlier_idxs
        vertical_plane_inlier_points = plane_1_inlier_points
        
    other_points = plane_2_remaining_points
    
    horizontal_z_val = np.mean(horizontal_plane_inlier_points, axis=0)[2]
    vertical_y_val = np.mean(vertical_plane_inlier_points, axis=0)[1]
    
    # Fill in the horizontal plane
    horizontal_projected_points = horizontal_plane_inlier_points[:, :2]
    hull = ConvexHull(horizontal_projected_points)
    polygon = Polygon(hull.points[hull.vertices])
    horizontal_sampled_points_2d = sample_points_within_polygon(polygon, num_sampled_points, sampling_method='uniform')
    horizontal_sampled_points_3d = np.hstack([horizontal_sampled_points_2d, np.full((horizontal_sampled_points_2d.shape[0], 1), horizontal_z_val)])
    
    # Fill in the vertical plane
    vertical_projected_points = vertical_plane_inlier_points[:, [0, 2]]
    hull = ConvexHull(vertical_projected_points)
    polygon = Polygon(hull.points[hull.vertices])
    vertical_sampled_points_2d = sample_points_within_polygon(polygon, num_sampled_points, sampling_method='uniform')
    vertical_sampled_points_3d = np.hstack([np.full((vertical_sampled_points_2d.shape[0], 1), vertical_y_val), vertical_sampled_points_2d])
    vertical_sampled_points_3d = vertical_sampled_points_3d[:, [1, 0, 2]]
    
    filled_shelf_points = np.vstack([horizontal_sampled_points_3d, vertical_sampled_points_3d, other_points])

    return filled_shelf_points


if __name__ == '__main__':
    visualize = False
    object_class = 'bottle'
    task = 'place'
    split = 'train'
    assert object_class in ['mug', 'bowl', 'bottle'], f'Invalid object class: {object_class}, must be either mug, bowl, or bottle.'
    assert task in ['grasp', 'place'], f'Invalid task: {task}, must be either grasp or place.'
    assert split in ['train', 'test'], f'Invalid split: {split}, must be either train or test.'
    
    save_augment = True
    savedir = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bottle_place_full_augment'
    
    data_root = f'/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bottle_place_with_bottoms/train_data/renders'
    data_root_files = glob.glob(os.path.join(data_root, '*.npz'))
    
    target_suffix = '_teleport_obj_points.npz' if task == 'place' else 'pre_grasp_obj_points.npz'
    object_files = [file for file in data_root_files if file.endswith(target_suffix)]

    print(f'Loaded {len(object_files)} files.')

    for file in object_files:
        shelf_data, other_data = load_shelf_point_cloud(file)
        
        augmented_pcd = fill_shelf_surface(shelf_data)
        
        if save_augment:
            # Save the augmented point cloud
            save_full_dir = os.path.join(savedir, f'{split}_data/renders')
            save_filled_shelf_point_cloud(file, save_full_dir, augmented_pcd)