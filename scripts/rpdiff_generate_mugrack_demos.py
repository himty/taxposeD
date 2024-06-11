import argparse
import datetime
import glob
import json
import numpy as np
import os
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import Transform3d, Rotate, Translate
import re
import torch
from typing import List, Tuple, Dict, Any

from equivariant_pose_graph.utils.env_mod_utils import combine_axis_aligned_rect, points_to_axis_aligned_rect, axis_aligned_rect_intersect
from equivariant_pose_graph.utils.error_metrics import matrix_from_list
from equivariant_pose_graph.utils.se3 import random_se3
from equivariant_pose_graph.utils.visualizations import plot_multi_np

def test_load_files():
    # demo_dir = '/home/odonca/workspace/rpad/rpdiff/src/rpdiff/data/task_demos/mug_rack_multi_mod_seed0/task_name_mug_on_rack_multi'
    demo_dir = '/home/odonca/workspace/rpad/rpdiff/src/rpdiff/data/task_demos/mug_rack_multi_mod_seed0/task_name_mug_on_rack_multi/syn_rack_med_3/1c9f9e25c654cbca3c71bf3f4dd78475'

    demo_files = glob.glob(demo_dir + '/*.npz')
    print(f'demo_files: {demo_files}')

    demo_data = np.load(demo_files[0], allow_pickle=True)
    print(f'demo_data: {demo_data.files}')

    start_obj_poses = demo_data['multi_obj_start_obj_pose']
    final_obj_poses = demo_data['multi_obj_final_obj_pose']
    start_obj_pcds = demo_data['multi_obj_start_pcd']
    final_obj_pcds = demo_data['multi_obj_final_pcd']

    # print(f'start_obj_poses: {start_obj_poses}')
    # print(f'final_obj_poses: {final_obj_poses}')
    # print(f'start_obj_pcds: {start_obj_pcds}')
    # print(f'final_obj_pcds: {final_obj_pcds}')

    child_start_obj_poses = start_obj_poses.item()['child']
    child_final_obj_poses = final_obj_poses.item()['child']
    child_start_obj_pcds = start_obj_pcds.item()['child']
    child_final_obj_pcds = final_obj_pcds.item()['child']

    parent_start_obj_poses = start_obj_poses.item()['parent']
    parent_final_obj_poses = final_obj_poses.item()['parent']
    parent_start_obj_pcds = start_obj_pcds.item()['parent']
    parent_final_obj_pcds = final_obj_pcds.item()['parent']

    print(f'child_start_obj_poses: {np.array(child_start_obj_poses).shape}')
    print(f'child_final_obj_poses: {np.array(child_final_obj_poses).shape}')
    print(f'child_start_obj_pcds: {child_start_obj_pcds.shape}')
    print(f'child_final_obj_pcds: {child_final_obj_pcds.shape}')

    print(f'parent_start_obj_poses: {np.array(parent_start_obj_poses).shape}')
    print(f'parent_final_obj_poses: {np.array(parent_final_obj_poses).shape}')
    print(f'parent_start_obj_pcds: {parent_start_obj_pcds.shape}')
    print(f'parent_final_obj_pcds: {parent_final_obj_pcds.shape}')

    child_start_pose_mat = matrix_from_list(child_start_obj_poses[0])
    child_final_pose_mat = matrix_from_list(child_final_obj_poses[0])

    print(f'child_start_pose_mat: {child_start_pose_mat}')
    print(f'child_final_pose_mat: {child_final_pose_mat}')

    child_start_pose_tf = Transform3d(matrix=torch.from_numpy(child_start_pose_mat.T))
    child_final_pose_tf = Transform3d(matrix=torch.from_numpy(child_final_pose_mat.T))

    print(f'child_start_pose_tf matrix: {child_start_pose_tf.get_matrix().transpose(-2, -1)}')

    child_start_obj_pcd_torch = torch.from_numpy(child_start_obj_pcds).double()
    child_final_obj_pcd_torch = torch.from_numpy(child_final_obj_pcds).double()

    child_start_to_final_tf = child_start_pose_tf.inverse().compose(child_final_pose_tf)
    child_start_to_final_pcd = child_start_to_final_tf.transform_points(child_start_obj_pcd_torch).detach().numpy()

    parent_start_pose_tf = Transform3d(matrix=torch.from_numpy(matrix_from_list(parent_start_obj_poses[0]).T))
    parent_final_pose_tf = Transform3d(matrix=torch.from_numpy(matrix_from_list(parent_final_obj_poses[0]).T))

    parent_start_obj_pcd_torch = torch.from_numpy(parent_start_obj_pcds).double()
    parent_final_obj_pcd_torch = torch.from_numpy(parent_final_obj_pcds).double()

    parent_start_to_origin_tf = parent_start_pose_tf.inverse()
    parent_start_to_final_tf = parent_start_to_origin_tf.compose(parent_final_pose_tf)

    parent_start_to_origin_pcd = parent_start_to_origin_tf.transform_points(parent_start_obj_pcd_torch).detach().numpy()

    child_origin_to_final_tf = child_final_pose_tf.compose(parent_start_to_origin_tf)
    child_start_to_origin_tf = child_start_pose_tf.inverse()

    child_origin_pcd = child_start_to_origin_tf.transform_points(child_start_obj_pcd_torch)
    child_origin_to_final_pcd = child_origin_to_final_tf.transform_points(child_origin_pcd)

    # plot_multi_np([
    #     child_start_obj_pcds, 
    #     parent_start_obj_pcds, 
    #     parent_final_obj_pcds, 
    #     child_start_to_final_pcd, 
    #     parent_start_to_origin_pcd, 
    #     child_origin_pcd.detach().numpy(), 
    #     child_origin_to_final_pcd.detach().numpy()
    # ])
    
    T0 = random_se3(
        1,
        rot_var=180 * np.pi / 180,
        trans_var=0.5,
        rot_sample_method="quat_uniform")
    T1 = random_se3(
        1, 
        rot_var=180 * np.pi / 180,
        trans_var=0.5,
        rot_sample_method="random_flat_upright")

    tf_action = T0.transform_points(child_final_obj_pcd_torch.float())
    tf_anchor = T1.transform_points(parent_final_obj_pcd_torch.float())

    gt_tf = T0.inverse().compose(T1)
    tf_action_gt = gt_tf.transform_points(tf_action)

    print(f'child_final_obj_pcd_torch: {child_final_obj_pcd_torch.shape}')
    print(f'parent_final_obj_pcd_torch: {parent_final_obj_pcd_torch.shape}')


    child_pcd_at_rack_origin_torch = parent_start_to_origin_tf.transform_points(child_final_obj_pcd_torch)
    parent_pcd_at_rack_origin_torch = parent_start_to_origin_tf.transform_points(parent_final_obj_pcd_torch)

    T_scene = random_se3(
        1,
        rot_var=180 * np.pi,
        trans_var=1,
        rot_sample_method="random_flat_upright"
    )

    child_pcd_scene = T_scene.transform_points(child_pcd_at_rack_origin_torch.float())
    parent_pcd_scene = T_scene.transform_points(parent_pcd_at_rack_origin_torch.float())
    
    # Get the transform from the final pose to the origin
    rack_final_to_origin_tf = Transform3d(
        matrix=torch.from_numpy(
            matrix_from_list(
                demo_data['multi_obj_final_obj_pose'].item()['parent'][0]
            ).T
        )
    ).inverse()
    
    # Get the final pose of the mug in the demo pose
    mug_final_pose = Transform3d(
        matrix=torch.from_numpy(
            matrix_from_list(
                demo_data['multi_obj_final_obj_pose'].item()['child'][0]
            ).T
        )
    )
    
    # Get the demo mug pose relative to the rack's origin
    rack_origin_to_mug_goal_tf = mug_final_pose.compose(rack_final_to_origin_tf)

    child_pcd_at_child_origin_torch = child_final_pose_tf.inverse().transform_points(child_final_obj_pcd_torch)
    child_pcd_origin_to_final_torch = rack_origin_to_mug_goal_tf.transform_points(child_pcd_at_child_origin_torch)

    action_points = child_pcd_scene - child_pcd_scene.mean(dim=0, keepdim=True)
    anchor_points = parent_pcd_scene - child_pcd_scene.mean(dim=0, keepdim=True)

    action_points_transformed = T0.transform_points(action_points.float())
    anchor_points_transformed = T1.transform_points(anchor_points.float())

    gt_tf = T0.inverse().compose(T1)
    tf_action_gt = gt_tf.transform_points(action_points_transformed)

    available_mug_poses = np.load('/home/odonca/workspace/rpad/data/rpdiff/data/descriptions/objects/syn_rack_med_unnormalized/available_mug_poses/syn_rack_med_3/1c9f9e25c654cbca3c71bf3f4dd78475/mug_poses.npz', allow_pickle=True)['mug_poses']
    print(f'available_mug_poses: {available_mug_poses.shape}')
    
    available_mug_poses_tf = Transform3d(matrix=torch.from_numpy(available_mug_poses).float().permute(0, 2, 1))
    child_pcd_at_child_origin_torch_tiled = child_pcd_at_child_origin_torch.unsqueeze(0).repeat(available_mug_poses.shape[0], 1, 1)
    
    avail_mug_pcds = available_mug_poses_tf.transform_points(child_pcd_at_child_origin_torch_tiled.float())
    avail_mug_pcds_list = [avail_mug_pcds[i] for i in range(avail_mug_pcds.shape[0])] 

    # Move origin rack to scene rack
    avail_mug_poses_scene = available_mug_poses_tf.compose(T_scene)

    # Move scene rack to action mean
    scene_to_child_mean_tf = Translate(-child_pcd_scene.mean(dim=0, keepdim=True).float())
    avail_mug_poses_scene_centered = avail_mug_poses_scene.compose(scene_to_child_mean_tf)
    
    # Move action mean rack to anchor pose
    avail_mug_poses_anchor = avail_mug_poses_scene_centered.compose(T1)
    
    avail_mug_pcds_at_anchor = avail_mug_poses_anchor.transform_points(child_pcd_at_child_origin_torch_tiled.float())
    avail_mug_pcds_list_at_anchor = [avail_mug_pcds_at_anchor[i] for i in range(avail_mug_pcds_at_anchor.shape[0])]

    plot_multi_np([
        # child_pcd_at_rack_origin_torch.detach().numpy(), 
        # parent_pcd_at_rack_origin_torch.detach().numpy(),
        # child_pcd_at_child_origin_torch.detach().numpy(),
        # child_pcd_origin_to_final_torch.detach().numpy(),
        child_pcd_scene.detach().numpy(),
        parent_pcd_scene.detach().numpy(),
        action_points.detach().numpy(),
        anchor_points.detach().numpy(),
        action_points_transformed.detach().numpy(),
        anchor_points_transformed.detach().numpy(),
        tf_action_gt.detach().numpy()
    ] + [mug_pcd.detach().numpy() for mug_pcd in avail_mug_pcds_list_at_anchor])

def downsample_pcd(points, num_points, type="fps"):
    if re.match(r"^fps$", type) is not None:
        return sample_farthest_points(points, K=num_points, random_start_point=True)
    elif re.match(r"^random$", type) is not None:
        random_idx = torch.randperm(points.shape[1])[:num_points]
        return points[:, random_idx], random_idx
    elif re.match(r"^random_0\.[0-9]$", type) is not None:
        prob = float(re.match(r"^random_(0\.[0-9])$", type).group(1))
        if np.random.random() > prob:
            return sample_farthest_points(points, K=num_points, random_start_point=True)
        else:
            random_idx = torch.randperm(points.shape[1])[:num_points]
            return points[:, random_idx], random_idx
    elif re.match(r"^[0-9]+N_random_fps$", type) is not None:
        random_num_points = int(re.match(r"^([0-9]+)N_random_fps$", type).group(1)) * num_points
        random_idx = torch.randperm(points.shape[1])[:random_num_points]
        random_points = points[:, random_idx]
        return sample_farthest_points(random_points, K=num_points, random_start_point=True)
    else:
        raise NotImplementedError(f"Downsample type {type} not implemented")


def get_available_mug_poses(demo_files):
    avail_mug_poses = [] # These will be relative to the rack's origin
    for demo_file in demo_files:
        demo_data = np.load(demo_file, allow_pickle=True)

        start_obj_poses = demo_data['multi_obj_start_obj_pose']
        final_obj_poses = demo_data['multi_obj_final_obj_pose']
        start_obj_pcds = demo_data['multi_obj_start_pcd']
        final_obj_pcds = demo_data['multi_obj_final_pcd']

        child_start_obj_poses = start_obj_poses.item()['child']
        child_final_obj_poses = final_obj_poses.item()['child']
        child_start_obj_pcds = start_obj_pcds.item()['child']
        child_final_obj_pcds = final_obj_pcds.item()['child']

        parent_start_obj_poses = start_obj_poses.item()['parent']
        parent_final_obj_poses = final_obj_poses.item()['parent']
        parent_start_obj_pcds = start_obj_pcds.item()['parent']
        parent_final_obj_pcds = final_obj_pcds.item()['parent']

        child_start_pose_mat = matrix_from_list(child_start_obj_poses[0])
        child_final_pose_mat = matrix_from_list(child_final_obj_poses[0])

        child_start_pose_tf = Transform3d(matrix=torch.from_numpy(child_start_pose_mat.T))
        child_final_pose_tf = Transform3d(matrix=torch.from_numpy(child_final_pose_mat.T))

        child_start_obj_pcd_torch = torch.from_numpy(child_start_obj_pcds).double()
        child_final_obj_pcd_torch = torch.from_numpy(child_final_obj_pcds).double()

        child_start_to_final_tf = child_start_pose_tf.inverse().compose(child_final_pose_tf)
        child_start_to_final_pcd = child_start_to_final_tf.transform_points(child_start_obj_pcd_torch).detach().numpy()

        parent_start_pose_tf = Transform3d(matrix=torch.from_numpy(matrix_from_list(parent_start_obj_poses[0]).T))
        parent_final_pose_tf = Transform3d(matrix=torch.from_numpy(matrix_from_list(parent_final_obj_poses[0]).T))

        parent_start_obj_pcd_torch = torch.from_numpy(parent_start_obj_pcds).double()
        parent_final_obj_pcd_torch = torch.from_numpy(parent_final_obj_pcds).double()

        parent_start_to_origin_tf = parent_start_pose_tf.inverse()
        parent_start_to_final_tf = parent_start_to_origin_tf.compose(parent_final_pose_tf)

        child_origin_to_final_tf = child_final_pose_tf.compose(parent_start_to_origin_tf)
        child_start_to_origin_tf = child_start_pose_tf.inverse()

        child_origin_pcd = child_start_to_origin_tf.transform_points(child_start_obj_pcd_torch)
        child_origin_to_final_pcd = child_origin_to_final_tf.transform_points(child_origin_pcd)
        
        avail_mug_poses.append(child_origin_to_final_tf)
    return avail_mug_poses


def get_non_intersecting_scene(anchor_pcd_list: List[torch.Tensor], action_pcd: torch.Tensor, action_anchor_map: int,
                               max_tries: int = 100) -> Tuple[List[torch.Tensor], torch.Tensor, List[Transform3d]]:
    """
    Given a list of anchor point clouds and an action point cloud, return K non-intersecting transforms such that applying each transform
    to the corresponding anchor point cloud (and potentially the action point cloud) results in a non-intersecting scene
    
    Args:
        anchor_pcd_list (List[torch.Tensor]): A list of anchor point cloud
        action_pcd (torch.Tensor): The action point cloud
        action_anchor_map (int): The index of which anchor point cloud the action point cloud belongs to
        max_tries (int): The maximum number of tries to find a non-intersecting scene
    
    Returns:
        if max_tries is reached: 
            None
        else:
            List[torch.Tensor]: A list of K non-intersecting point clouds
            torch.Tensor: The transformed action point cloud
            List[Transform3d]: A list of K non-intersecting transforms
    """
    # Prepare the point clouds, potentially combining the action point cloud with an anchor point cloud
    pcd_list = []
    for i, anchor_pcd in enumerate(anchor_pcd_list):
        if i == action_anchor_map:
            pcd_list.append(torch.cat([anchor_pcd, action_pcd], dim=0))
        else:
            pcd_list.append(anchor_pcd)
    
    max_tries = 50
    while max_tries > 0:
        # Sample random SE3 transforms
        se3_list = random_se3(
            len(pcd_list),
            rot_var = 180 * np.pi / 180,
            trans_var = 1.0,
            rot_sample_method="random_flat_upright"
        )
    
        transformed_pcd_list = []
        transformed_pcd_rects = []
        for i in range(len(pcd_list)):
            transformed_pcd = se3_list[i].transform_points(pcd_list[i].float())
            transformed_pcd_list.append(transformed_pcd)
            transformed_pcd_rects.append(points_to_axis_aligned_rect(transformed_pcd.unsqueeze(0)))
            
        # Check if the transformed point clouds are non-intersecting
        # TODO: Maybe replace this by actually loading the mesh in sim to get better intersection checks
        intersects = False
        scene_rect = transformed_pcd_rects[0]
        for i in range(1, len(transformed_pcd_rects)):
            new_rect = transformed_pcd_rects[i]
            
            cur_intersects = axis_aligned_rect_intersect(scene_rect, new_rect)[0]
            if cur_intersects:
                intersects = True
                break
            else:
                scene_rect = combine_axis_aligned_rect([scene_rect, new_rect])
        
        if not intersects:
            break
            
        max_tries -= 1
    
    if max_tries == 0:
        return None
    else:
        out_transform_list = []
        out_pcd_list = []
        out_action_pcd = None
        for i in range(len(pcd_list)):
            out_transform_list.append(se3_list[i].get_matrix().transpose(-2, -1))
            if i == action_anchor_map:
                out_pcd_list.append(transformed_pcd_list[i][:-action_pcd.shape[0]])
                out_action_pcd = transformed_pcd_list[i][-action_pcd.shape[0]:]
            else:
                out_pcd_list.append(transformed_pcd_list[i])
        return out_pcd_list, out_action_pcd, out_transform_list
            
    
def create_rack_scene(rack_mug_demos: Dict[str, Any], valid_racks: List[str], mug_id: str, chosen_racks: List[str], descriptions_dir) -> Dict[str, Any]:
    """
    Given a dictionary of rack-mug demos, create a scene of K racks with the target mug
    
    Args:
        rack_mug_demos (Dict[str, Any]): A dictionary of rack-mug demos
        mug_id (str): The target mug id
        chosen_racks (List[str]): The list of chosen racks
        descriptions_dir (str): The directory containing the descriptions
        
    Returns:
        if no non-intersecting scene is found:
            None
        else:
            Dict[str, Any]: A dictionary containing scene information
    """
    
    scene_dict = {
        'ids': {
            'mug': mug_id,
            'racks': chosen_racks
        },
    }
    
    # Select random demo to get mug placement
    all_possible_demos = []
    # all_avail_poses = []
    for rack in chosen_racks:
        print(f'rack: {rack}, mug_id: {mug_id}, num demos: {len(rack_mug_demos[rack][mug_id])}')
        all_possible_demos.extend(rack_mug_demos[rack][mug_id])
        # rack_type = re.match(r"^(syn_rack_[a-zA-Z]+)_[0-9-]+$", rack).group(1)
        # rack_avail_mug_poses = np.load(
        #     os.path.join(descriptions_dir, f'objects/{rack_type}_unnormalized/available_mug_poses/{rack}/{mug_id}/mug_poses.npz'),
        #     allow_pickle=True
        # )['mug_poses']
        # all_avail_poses.append(rack_avail_mug_poses)

    scene_dict['available_demos'] = {'available_demos': all_possible_demos}
    # scene_dict['available_mug_poses'] = {'available_mug_poses': all_avail_poses}
    
    random_demo = np.random.choice(list(all_possible_demos))
    random_demo_rack = random_demo.split('/')[-3]
    scene_dict['selected_demo'] = {'selected_demo': random_demo}
    scene_dict['selected_demo_rack'] = {'selected_demo_rack': random_demo_rack}
    
    
    # Get the point clouds for the racks and the mug
    demo_rack_pcds = []
    demo_rack_final_poses = []
    demo_mug_pcd = None
    demo_mug_final_pose = None
    mug_rack_map = 0
    for i, rack in enumerate(chosen_racks):
        if rack == random_demo_rack:
            # Load the specific demo to get the mug and rack pcd
            demo_data = np.load(random_demo, allow_pickle=True)
            
            # Get the point clouds in the final demo pose            
            final_obj_pcds = demo_data['multi_obj_final_pcd'].item()
            final_obj_poses = demo_data['multi_obj_final_obj_pose'].item()
            
            rack_pcd = final_obj_pcds['parent']
            rack_pose = final_obj_poses['parent']
            mug_pcd = final_obj_pcds['child']
            mug_pose = final_obj_poses['child']
            
            # Get the transform from the final pose to the origin
            rack_final_to_origin_tf = Transform3d(
                matrix=torch.from_numpy(
                    matrix_from_list(
                        demo_data['multi_obj_final_obj_pose'].item()['parent'][0]
                    ).T
                )
            ).inverse()
            
            # Move both the rack and mug point clouds to the origin
            rack_pcd_origin = rack_final_to_origin_tf.transform_points(torch.from_numpy(rack_pcd))
            mug_pcd_origin = rack_final_to_origin_tf.transform_points(torch.from_numpy(mug_pcd))

            demo_rack_pcds.append(rack_pcd_origin)
            demo_rack_final_poses.append(rack_pose)
            demo_mug_pcd = mug_pcd_origin
            demo_mug_final_pose = mug_pose
            mug_rack_map = i
        else:
            # Just load any demo of the rack to get the associated rack pcd
            demo_data = np.load(np.random.choice(rack_mug_demos[rack][mug_id]), allow_pickle=True)

            # Get the point clouds in the final demo pose            
            final_obj_pcds = demo_data['multi_obj_final_pcd'].item()
            final_obj_poses = demo_data['multi_obj_final_obj_pose'].item()
            rack_pcd = final_obj_pcds['parent']
            rack_pose = final_obj_poses['parent']
            
            # Get the transform from the final pose to the origin
            rack_final_to_origin_tf = Transform3d(
                matrix=torch.from_numpy(
                    matrix_from_list(
                        demo_data['multi_obj_final_obj_pose'].item()['parent'][0]
                    ).T
                )
            ).inverse()
            
            # Move both the rack and mug point clouds to the origin
            rack_pcd_origin = rack_final_to_origin_tf.transform_points(torch.from_numpy(rack_pcd))

            demo_rack_pcds.append(rack_pcd_origin)
            demo_rack_final_poses.append(rack_pose)

    scene_dict['final_pcds'] = {'racks': demo_rack_pcds, 'mug': demo_mug_pcd}
    scene_dict['final_poses'] = {'racks': demo_rack_final_poses, 'mug': demo_mug_final_pose}
    scene_dict['demo_rack_idx'] = {'demo_rack_idx': mug_rack_map}  
    
    # Create a scene with multiple racks
    out = get_non_intersecting_scene(demo_rack_pcds, demo_mug_pcd, mug_rack_map)
    if out is None:
        print('Could not find non-intersecting scene')
        return None
    non_intersecting_rack_pcds, transformed_demo_mug, non_intersecting_rack_transforms = out

    scene_dict['scene_pcds'] = {'racks': non_intersecting_rack_pcds, 'mug': transformed_demo_mug}
    scene_dict['scene_rack_transforms'] = {'racks': non_intersecting_rack_transforms}
    
    return scene_dict


def save_scene_dict(save_dir: str, iteration: int, scene_dict: Dict[str, Any], save_dict_padding: int = 0, save_minimal: bool = False, training: bool = True) -> None:
    """
    Save the scene dictionary to a file
    
    Args:
        save_dir (str): The directory to save the file
        iteration (int): The iteration number
        scene_dict (Dict[str, Any]): The scene dictionary
        save_dict_padding (int): The padding to add to each dict element to support PyTorch batching
        save_minimal (bool): If True, only save the minimal information for the scene
        training (bool): If True, the scene is a training scene, otherwise it is a validation scene
    """
    # points_action = scene_dict['scene_mug_pcd']
    # points_anchor = torch.cat(scene_dict['scene_rack_pcds'], dim=0)
    
    points_action = scene_dict['scene_pcds']['mug']
    points_anchor = torch.cat(scene_dict['scene_pcds']['racks'], dim=0)
    
    clouds = torch.cat([points_anchor, points_action], dim=0)
    classes = torch.cat([torch.ones(points_anchor.shape[0]), torch.zeros(points_action.shape[0])], dim=0)

    print(f'points_action: {points_action.shape}')
    print(f'points_anchor: {points_anchor.shape}')

    if save_minimal:
        if save_dict_padding == 0:
            scene_save_dict = {
                'ids': scene_dict['ids'],
                'final_poses': scene_dict['final_poses'],
                'scene_rack_transforms': scene_dict['scene_rack_transforms'],
                'demo_rack_idx': scene_dict['demo_rack_idx']
            }
        else:
            padded_ids = {
                'mug': scene_dict['ids']['mug'],
                'racks': scene_dict['ids']['racks'] + ['']*(save_dict_padding - len(scene_dict['ids']['racks']))
            }
            
            final_poses_rack_np = np.array(scene_dict['final_poses']['racks'])
            padding_final_poses_racks = np.full((save_dict_padding - final_poses_rack_np.shape[0], final_poses_rack_np.shape[1], final_poses_rack_np.shape[2]), np.nan)
            padded_final_poses_racks = np.concatenate([final_poses_rack_np, padding_final_poses_racks], axis=0)
            
            padded_scene_rack_transforms = scene_dict['scene_rack_transforms']['racks'] +\
                [torch.full(scene_dict['scene_rack_transforms']['racks'][0].shape, np.nan)]*\
                    (save_dict_padding - len(scene_dict['scene_rack_transforms']['racks']))
            
            scene_save_dict = {
                'ids': padded_ids,
                'final_poses': {
                    'racks': padded_final_poses_racks,
                    'mug': scene_dict['final_poses']['mug']
                },
                'scene_rack_transforms': {
                    'racks': padded_scene_rack_transforms
                },
                'demo_rack_idx': scene_dict['demo_rack_idx']
            }
    else:
        if save_dict_padding == 0:
            scene_save_dict = scene_dict
        else:
            raise NotImplementedError('Padding not implemented for full scene dict')

    save_dict = {
        'clouds': clouds,
        'classes': classes,
        'colors': None,
        'shapenet_id': None,
        **scene_save_dict
    }
    
    print(f'save_dict: {save_dict.keys()}')
    train_val = 'train' if training else 'val'
    np.savez_compressed(f'{save_dir}/{train_val}/{iteration}_teleport_obj_points.npz', **save_dict)


def create_demos(demo_dir: str, save_dir: str, descriptions_dir: str, 
                 max_num_racks: int, max_num_mugs: int, max_num_demos: int, train_ratio: float = 0.8,
                 K: int = 3, up_to_K: bool = False, up_to_k_probs: List[float] = None, 
                 min_demos: int = 1, save_minimal: bool = False, save_dict_padding: int = 0) -> None:
    """
    Create a set of non-intersecting scenes from the given demos
    
    Args:
        demo_dir (str): The directory containing the demos
        save_dir (str): The directory to save the scenes
        descriptions_dir (str): The directory containing the descriptions
        max_num_racks (int): The maximum number of racks to create scenes with
        max_num_mugs (int): The maximum number of mugs to create scenes with
        max_num_demos (int): The number of scenes to create
        train_ratio (float): The ratio of training to validation scenes
        K (int): The number of racks to create the scene with
        up_to_K (bool): If True, the scene will have up to K racks, otherwise exactly K racks
        up_to_k_probs (List[float]): The probabilities of having up to K racks
        min_demos (int): The minimum number of demos for a rack-mug pair to be considered
        save_minimal (bool): If True, only save the minimal information for the scene
        save_dict_padding (int): The padding to add to each dict element to support PyTorch batching
    """

    all_racks = glob.glob(demo_dir + '/syn_*')
    unique_racks = set([rack.split('/')[-1] for rack in all_racks])

    all_mug_ids = glob.glob(demo_dir + '/syn_*/*')
    unique_mug_ids = set([mug_id.split('/')[-1] for mug_id in all_mug_ids])

    print(f'unique_racks: {unique_racks}')
    print(f'unique_mug_ids: {unique_mug_ids}')
    print('-'*50)

    # Get the demos for each rack-mug pair
    rack_mug_demos = {}
    for rack in unique_racks:
        rack_mug_demos[rack] = {}
        for mug_id in unique_mug_ids:
            rack_mug_demos[rack][mug_id] = glob.glob(demo_dir + f'/{rack}/{mug_id}/*.npz')
    
    start_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset_save_dir = os.path.join(save_dir, f'{start_datetime}')
    os.makedirs(os.path.join(dataset_save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_save_dir, 'val'), exist_ok=True)
    dataset_dict = {
        'used_racks': {},
        'used_mugs': {},
        'k_distribution': {},
        'train_count': 0,
        'val_count': 0,
        'total_count': 0,
    }
    
    demo_iter = 0
    while dataset_dict['train_count'] < max_num_demos and dataset_dict['val_count'] < max_num_demos:
        print(f'Creating demo {demo_iter+1}')
        # Choose a random mug
        if len(dataset_dict['used_mugs'].keys()) < max_num_mugs:
            selected_mug_id = np.random.choice(list(unique_mug_ids))
        else:
            selected_mug_id = np.random.choice(list(dataset_dict['used_mugs'].keys()))

        # Check that we have at least K racks which have at least min_demos demos for the selected mug
        valid_racks = []
        for rack in unique_racks:
            if len(rack_mug_demos[rack][selected_mug_id]) >= min_demos:
                valid_racks.append(rack)
        if len(valid_racks) < K:
            print(f'Not enough racks with at least {min_demos} demos for the selected mug')
            continue

        print(f'\tmug_id: {selected_mug_id}')
        print(f'\tvalid_racks: {valid_racks}')

        # First create a scene of K racks
        chosen_racks = []
        if up_to_K:
            num_racks_ = np.arange(1, K+1)
            if up_to_k_probs is None:
                num_racks = np.random.choice(num_racks_)
            else:
                num_racks = np.random.choice(num_racks_, p=up_to_k_probs)
        else:
            num_racks = K
        
        failed_attempt = False
        for i in range(num_racks):
            if len(dataset_dict['used_racks'].keys()) < max_num_racks:
                chosen_rack = np.random.choice(valid_racks)
            else:
                num_tries = 10
                while num_tries > 0:
                    chosen_rack = np.random.choice(list(dataset_dict['used_racks'].keys()))
                    valid_selection = True
                    for rack in chosen_racks:
                        if len(rack_mug_demos[chosen_rack][selected_mug_id]) < min_demos:
                            valid_selection = False
                    if valid_selection:
                        break
                    
                    num_tries -= 1
                if num_tries == 0:
                    failed_attempt = True
                            
            chosen_racks.append(chosen_rack)

        if failed_attempt:
            continue

        try:
            scene_dict = create_rack_scene(rack_mug_demos, valid_racks, selected_mug_id, chosen_racks, descriptions_dir)
            if scene_dict is None:
                print('Could not find non-intersecting scene')
                continue
        except Exception as e:
            print(f'Error creating scene: {e}')
            continue
        
        training = np.random.random() < train_ratio
        save_scene_dict(
            dataset_save_dir, 
            demo_iter, 
            scene_dict, 
            save_dict_padding=save_dict_padding, 
            save_minimal=save_minimal,
            training=training
        )

        dataset_dict['k_distribution'][int(num_racks)] = dataset_dict['k_distribution'].get(num_racks, 0) + 1
        dataset_dict['used_mugs'][selected_mug_id] = dataset_dict['used_mugs'].get(selected_mug_id, 0) + 1
        for chosen_rack in chosen_racks:
            dataset_dict['used_racks'][chosen_rack] = dataset_dict['used_racks'].get(chosen_rack, 0) + 1

        dataset_dict['total_count'] += 1        
        if training:
            dataset_dict['train_count'] += 1
        else:
            dataset_dict['val_count'] += 1
        
        demo_iter += 1

    dataset_dict['num_unique_racks'] = len(dataset_dict['used_racks'].keys())
    dataset_dict['num_unique_mugs'] = len(dataset_dict['used_mugs'].keys())
    
    print(f'dataset_dict: {dataset_dict}')
    with open(os.path.join(dataset_save_dir, 'dataset_dict.json'), 'w') as f:
        json.dump(dataset_dict, f)


def create_descriptions(demo_dir: str, descriptions_dir: str):
    all_racks = glob.glob(demo_dir + '/syn_*')
    unique_racks = set([rack.split('/')[-1] for rack in all_racks])

    all_mug_ids = glob.glob(demo_dir + '/syn_*/*')
    unique_mug_ids = set([mug_id.split('/')[-1] for mug_id in all_mug_ids])

    print(f'unique_racks: {unique_racks}')
    print(f'unique_mug_ids: {unique_mug_ids}')
    print('-'*50)

    mug_rack_demos = {}
    for rack in unique_racks:
        mug_rack_demos[rack] = {}
        for mug in unique_mug_ids:
            print(f'rack: {rack}, mug: {mug}')
            demos = glob.glob(demo_dir + f'/{rack}/{mug}/*.npz')
            print(f'\tDemos: {len(demos)}')
            
            mug_poses = []
            for demo in demos:
                # Load the specific demo to get the mug and rack pcd
                demo_data = np.load(demo, allow_pickle=True)
                
                # Get the transform from the final pose to the origin
                rack_final_to_origin_tf = Transform3d(
                    matrix=torch.from_numpy(
                        matrix_from_list(
                            demo_data['multi_obj_final_obj_pose'].item()['parent'][0]
                        ).T
                    )
                ).inverse()
                
                # Get the final pose of the mug in the demo pose
                mug_final_pose = Transform3d(
                    matrix=torch.from_numpy(
                        matrix_from_list(
                            demo_data['multi_obj_final_obj_pose'].item()['child'][0]
                        ).T
                    )
                )
                
                # Get the demo mug pose relative to the rack's origin
                rack_origin_to_mug_goal_tf = mug_final_pose.compose(rack_final_to_origin_tf)
            
                mug_poses.append(rack_origin_to_mug_goal_tf)
            
            if len(mug_poses) == 0:
                continue  
            mug_poses = torch.cat([pose.get_matrix().transpose(-2, -1) for pose in mug_poses], dim=0)
            
            mug_rack_demos[rack][mug] = mug_poses
            
            
    for rack in mug_rack_demos.keys():
        print(f'rack: {rack}')
        rack_type = re.match(r"^(syn_rack_[a-zA-Z]+)_[0-9-]+$", rack).group(1)
        rack_avail_mug_poses_dir = os.path.join(descriptions_dir, f'objects/{rack_type}_unnormalized/available_mug_poses/{rack}')
        os.makedirs(rack_avail_mug_poses_dir, exist_ok=True)
        for mug in mug_rack_demos[rack].keys():
            print(f'\tmug: {mug}, poses: {mug_rack_demos[rack][mug].shape}')
            os.makedirs(os.path.join(rack_avail_mug_poses_dir, mug), exist_ok=True)
            np.savez_compressed(os.path.join(rack_avail_mug_poses_dir, mug, 'mug_poses.npz'), mug_poses=mug_rack_demos[rack][mug].numpy())
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--create_demos", action="store_true")
    parser.add_argument("--create_descriptions", action="store_true")
    parser.add_argument("--demo_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--descriptions_dir", type=str, default=None)
    parser.add_argument("--max_num_demos", type=int, default=20)
    parser.add_argument("--max_num_racks", type=int, default=10)
    parser.add_argument("--max_num_mugs", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--up_to_k", action="store_true")
    parser.add_argument("--up_to_k_probs", type=float, nargs='+', default=None)
    parser.add_argument("--min_demos", type=int, default=1)
    parser.add_argument("--save_minimal", action="store_true")
    parser.add_argument("--save_dict_padding", type=int, default=0)
    
    args = parser.parse_args()
    
    # demo_dir = '/home/odonca/workspace/rpad/rpdiff/src/rpdiff/data/task_demos/mug_rack_multi_mod_seed0/task_name_mug_on_rack_multi'
    # save_dir = '/home/odonca/workspace/rpad/data/rpdiff/custom_mug_rack_data'
    # `/home/odonca/workspace/rpad/data/rpdiff/data/descriptions/objects/syn_rack_med_unnormalized/available_mug_poses`
    # num_demos = 20
    # K = 3
    # up_to_K = False
    # min_demos = 5

    # test_load_files()
    if args.test:
        test_load_files()
    elif args.create_demos:
        save_dir = args.save_dir
        if args.dataset_name is not None:
            save_dir = os.path.join(save_dir, args.dataset_name)
        create_demos(
            demo_dir=args.demo_dir,
            save_dir=save_dir,
            descriptions_dir=args.descriptions_dir,
            max_num_demos=args.max_num_demos, 
            max_num_racks=args.max_num_racks,
            max_num_mugs=args.max_num_mugs,
            train_ratio=args.train_ratio,
            K=args.K, 
            up_to_K=args.up_to_k,
            up_to_k_probs=args.up_to_k_probs,
            min_demos=args.min_demos,
            save_minimal=args.save_minimal,
            save_dict_padding=args.save_dict_padding
        )
    elif args.create_descriptions:
        demo_dir = args.demo_dir
        descriptions_dir = args.descriptions_dir
        create_descriptions(demo_dir, descriptions_dir)
        