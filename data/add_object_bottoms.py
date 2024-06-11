import sys
import os
import glob
import pathlib
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


def load_mug_point_cloud(filepath):
    """
    Load mug point cloud and remaining points from given file path.
    
    Parameters:
    - filepath: file path to .npz file containing point cloud data.
    
    Returns:
    - mug point cloud.
    - remaining point cloud.
    """
    data = np.load(filepath)
    points = data['clouds']
    classes = data['classes']
    
    # Get the mug points
    mug_points = points[classes == 0]
    other_points = points[classes != 0]
    
    return mug_points, other_points


def save_augmented_mug_point_cloud(filepath, savedir, augmented_pcd):
    """
    Save augmented mug point cloud to given file path.
    
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
    mug_idxs = np.where(classes == 0)[0]
    clouds = np.delete(clouds, mug_idxs, axis=0)
    classes = np.delete(classes, mug_idxs, axis=0)
    colors = np.delete(colors, mug_idxs, axis=0)
    
    # Add the augmented mug points and corresponding classes and colors
    clouds = np.vstack([clouds, augmented_pcd])
    classes = np.hstack([classes, np.full(augmented_pcd.shape[0], 0)])
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


def augment_mug_bottom(pcd, num_sampled_points, clipping_height, z_offset, rotation_matrix=None):
    """
    Augment the mug point cloud by sampling points within the convex hull of its bottom.
    
    Parameters:
    - pcd: Point cloud data for the mug.
    - num_sampled_points: Number of points to sample.
    - clipping_height: Height to determine which points represent the mug's bottom.
    - z_offset: Offset to adjust the z value of the sampled points.
    - rotation_matrix: Rotation matrix to apply to the augmented point cloud (optional).
    
    Returns:
    - Augmented point cloud.
    - Convex hull of the projected point cloud.
    - Projected 2D points.
    - Sampled 2D points.
    """
    # Apply rotation if it was provided
    if rotation_matrix is not None:
        pcd = rotation_matrix.apply(pcd)
    
    min_z = np.min(pcd[:, 2])
    
    # Keep points within [min_z, min_z + clipping_height]
    clipped_pcd = pcd[pcd[:, 2] <= min_z + clipping_height]
    projected_pcd_2d = clipped_pcd[:, :2]
    
    # Compute convex hull and sample points
    hull = ConvexHull(projected_pcd_2d)
    polygon = Polygon(hull.points[hull.vertices])
    sampled_points_2d = sample_points_within_polygon(polygon, num_sampled_points, sampling_method='uniform')
    sampled_points_3d = np.hstack([sampled_points_2d, np.full((sampled_points_2d.shape[0], 1), min_z + z_offset)])
    
    # Augment the original point cloud with the sampled points
    augmented_pcd = np.vstack([pcd, sampled_points_3d])
    
    # Undo rotation if it was provided
    if rotation_matrix is not None:
        augmented_pcd = rotation_matrix.inv().apply(augmented_pcd)
    
    return augmented_pcd, hull, projected_pcd_2d, sampled_points_2d


if __name__ == '__main__':
    visualize = True
    save_augment = False
    object_class = 'bowl'
    task = 'grasp'
    split = 'train'
    assert object_class in ['mug', 'bowl', 'bottle'], f'Invalid object class: {object_class}, must be either mug, bowl, or bottle.'
    assert task in ['grasp', 'place'], f'Invalid task: {task}, must be either grasp or place.'
    assert split in ['train', 'test'], f'Invalid split: {split}, must be either train or test.'

    use_special_cases = True
    if object_class == 'mug' and task == 'grasp':
        save_dir = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/mug_grasp_with_bottoms'

        mug_grasp_root = f'/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/{object_class}_{task}/{split}_data/renders'
        mug_grasp_root_files = glob.glob(os.path.join(mug_grasp_root, '*.npz'))
        pre_grasp_obj_files = [file for file in mug_grasp_root_files if file.endswith('pre_grasp_obj_points.npz')]
        
        # Parameters for augmentation
        clipping_height = 0.012
        num_sampled_points = 3000

    elif object_class == 'bottle' and task == 'grasp' and split == 'train':
        save_dir = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bottle_grasp_with_bottoms'
        
        mug_grasp_root = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bottle_grasp/bottle_train_new_3_pregrasp/renders'
        mug_grasp_root_files = glob.glob(os.path.join(mug_grasp_root, '*.npz'))
        pre_grasp_obj_files = [file for file in mug_grasp_root_files if file.endswith('pre_grasp_obj_points.npz')]
    
        # Parameters for augmentation
        clipping_height = 0.005
        num_sampled_points = 1500
        
    elif object_class == 'bottle' and task == 'place' and split == 'train':
        save_dir = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bottle_place_with_bottoms'
        
        mug_grasp_root = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bottle_place/bottle_train_data_ndf_cons_3/renders'
        mug_grasp_root_files = glob.glob(os.path.join(mug_grasp_root, '*.npz'))
        pre_grasp_obj_files = [file for file in mug_grasp_root_files if file.endswith('_teleport_obj_points.npz')]
    
        # Parameters for augmentation
        clipping_height = 0.005
        num_sampled_points = 1500
        
    elif object_class == 'bowl' and task == 'place' and split == 'train':
        save_dir = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bowl_place_with_bottoms'
        
        mug_grasp_root = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bowl_place/bowl_train_data_ndf_cons_3/renders'
        mug_grasp_root_files = glob.glob(os.path.join(mug_grasp_root, '*.npz'))
        pre_grasp_obj_files = [file for file in mug_grasp_root_files if file.endswith('_teleport_obj_points.npz')]
        
        # Parameters for augmentation
        clipping_height = 0.005
        num_sampled_points = 1500
        
    elif object_class == 'bowl' and task == 'grasp' and split == 'train':
        save_dir = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bowl_grasp_with_bottoms'
        
        mug_grasp_root = '/home/odonca/workspace/rpad/data/equivariant_pose_graph/data/bowl_grasp/bowl_train_new_3_pregrasp/renders'
        mug_grasp_root_files = glob.glob(os.path.join(mug_grasp_root, '*.npz'))
        pre_grasp_obj_files = [file for file in mug_grasp_root_files if file.endswith('pre_grasp_obj_points.npz')]
        
        # Parameters for augmentation
        clipping_height = 0.005
        num_sampled_points = 1000
    
    else:
        raise ValueError(f'Invalid combination of object class {object_class}, task {task}, and split {split}.')
    
    print(f'Number of pre-grasp obj files: {len(pre_grasp_obj_files)}')

    for idx, file_path in enumerate(pre_grasp_obj_files):
        # Load the point cloud for the current file
        pcd, other_pcd = load_mug_point_cloud(file_path)

        z_offset = 0
        rotation_matrix = None
        cur_clipping_height = clipping_height
        
        if use_special_cases:
            if object_class == 'mug' and task == 'grasp':
                if idx == 5: z_offset = 0.008
                if idx == 7: rotation_matrix = R.from_euler('xyz', [0, 6, 0], degrees=True)

            elif object_class == 'bottle' and task == 'grasp':
                if idx == 2: rotation_matrix = R.from_euler('xyz', [-2, 0, 0], degrees=True)
                if idx == 2: z_offset = 0.0005
                if idx == 6: rotation_matrix = R.from_euler('xyz', [-1, 0, 0], degrees=True)
                if idx == 6: z_offset = 0.002
            
            elif object_class == 'bottle' and task == 'place':
                if idx == 0: rotation_matrix = R.from_euler('xyz', [0, -1, 0], degrees=True)
                if idx == 0: z_offset = 0.0005
                if idx == 9: rotation_matrix = R.from_euler('xyz', [-2, 1, 0], degrees=True)
                if idx == 9: z_offset = 0.0005
                
            elif object_class == 'bowl' and task == 'grasp':
                if idx == 0: rotation_matrix = R.from_euler('xyz', [-4, -3, 0], degrees=True)
                if idx == 0: cur_clipping_height = 0.003
                if idx == 8: cur_clipping_height = 0.01
            
            else:
                raise ValueError(f'Invalid combination of object class {object_class}, task {task}, and split {split}.') 


        # Augment the point cloud
        augmented_pcd, hull, projected_pcd_2d, sampled_points_2d = augment_mug_bottom(pcd, num_sampled_points, cur_clipping_height, z_offset, rotation_matrix)

        if visualize:
            # Optional: Plot the original point cloud
            print(f'Plotting original pointcloud {idx}')
            # plot_multi_np([pcd])
            
            # Plot the projected and sampled points
            plt.figure(figsize=(8, 8))
            plt.scatter(projected_pcd_2d[:, 0], projected_pcd_2d[:, 1], alpha=0.6, label='Projected Points', s=1)
            plt.scatter(sampled_points_2d[:, 0], sampled_points_2d[:, 1], alpha=0.6, label='Sampled Points', s=1)
            for simplex in hull.simplices:
                plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')
            plt.xlim(np.min(projected_pcd_2d[:, 0]) - 0.05, np.max(projected_pcd_2d[:, 0]) + 0.05)
            plt.ylim(np.min(projected_pcd_2d[:, 1]) - 0.05, np.max(projected_pcd_2d[:, 1]) + 0.05)
            plt.legend()
            plt.title('Projected and Sampled Points')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)

            # Plot the augmented point cloud
            print(f'Plotting augmented pointcloud {idx}')
            
            # shift the original pcd to the right to visualize both of them together
            shifted_pcd = pcd.copy()
            shifted_pcd[:, 0] += [pcd[:, 0].max() - pcd[:, 0].min() + 0.05] * pcd.shape[0]
            
            plot_multi_np([augmented_pcd, shifted_pcd])
            
            plt.show()
        
        
        if save_augment:
            # Save the augmented point cloud
            save_full_dir = os.path.join(save_dir, f'{split}_data/renders')
            save_augmented_mug_point_cloud(file_path, save_full_dir, augmented_pcd)
