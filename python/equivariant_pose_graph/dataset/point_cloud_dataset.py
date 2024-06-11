from pathlib import Path
from turtle import down, st
import torch.nn.functional as F

import numpy as np
import os

import torch
from torch.utils.data import Dataset
# random_se3_alternative
from equivariant_pose_graph.utils.se3 import random_se3
from equivariant_pose_graph.utils.occlusion_utils import ball_occlusion, plane_occlusion, bottom_surface_occlusion
from pytorch3d.ops import sample_farthest_points
from torch_geometric.nn import fps
from pytorch3d.transforms import Rotate

from equivariant_pose_graph.utils.env_mod_utils import get_random_rack_demo

class PointCloudDataset(Dataset):
    def __init__(self, dataset_root,
                 dataset_indices=[10],
                 cloud_type='final',
                 action_class=0,
                 anchor_class=1,
                 num_points=1024,
                 dataset_size=1000,
                 rotation_variance=np.pi,
                 translation_variance=0.5,
                 symmetric_class=None,
                 angle_degree=180,
                 overfit=False,
                 overfit_distractor_aug=False,
                 num_overfit_transforms=3,
                 seed_overfit_transforms=False,
                 synthetic_occlusion=False,
                 ball_radius=None,
                 plane_standoff=None,
                 bottom_surface_z_clipping_height=0.1,
                 scale_point_clouds=False,
                 scale_point_clouds_min=0.5,
                 scale_point_clouds_max=2.0,
                 distractor_anchor_aug=False,
                 num_demo=12,
                 min_num_cameras = 4,
                 max_num_cameras = 4,
                 random_files=True,
                 demo_mod_k_range=[2, 2],
                 demo_mod_rot_var=np.pi/180 * 360,
                 demo_mod_trans_var=0.15,
                 action_rot_sample_method="axis_angle",
                 anchor_rot_sample_method="axis_angle",
                 distractor_rot_sample_method="axis_angle",
                 action_plane_occlusion=True,
                 action_ball_occlusion=True,
                 action_bottom_surface_occlusion=True,
                 anchor_plane_occlusion=False,
                 anchor_ball_occlusion=False,
                 anchor_bottom_surface_occlusion=False,
                 downsample_type="fps",
                 gaussian_noise_mu=0,
                 gaussian_noise_std=0.001,
                 ):
        self.dataset_size = dataset_size
        self.num_points = num_points
        self.dataset_root = Path(dataset_root)
        self.cloud_type = cloud_type
        self.rot_var = rotation_variance
        self.trans_var = translation_variance
        self.action_class = action_class
        self.anchor_class = anchor_class
        self.symmetric_class = symmetric_class  # None if no symmetric class exists
        self.angle_degree = angle_degree
        self.min_num_cameras = min_num_cameras
        self.max_num_cameras = max_num_cameras
        self.random_files = random_files

        self.overfit = overfit
        self.overfit_distactor_aug = overfit_distractor_aug
        self.seed_overfit_transforms = seed_overfit_transforms
        self.dataset_indices = dataset_indices
        self.num_overfit_transforms = num_overfit_transforms
        self.T0_list = []
        self.T1_list = []
        if self.dataset_indices == 'None':
            dataset_indices = self.get_existing_data_indices()
            self.dataset_indices = dataset_indices
        self.bad_demo_id = self.go_through_list()
        
        self.synthetic_occlusion = synthetic_occlusion
        self.ball_radius = ball_radius
        self.plane_standoff = plane_standoff
        self.bottom_surface_z_clipping_height = bottom_surface_z_clipping_height
        
        self.action_plane_occlusion = action_plane_occlusion
        self.action_ball_occlusion = action_ball_occlusion
        self.action_bottom_surface_occlusion = action_bottom_surface_occlusion
        self.anchor_plane_occlusion = anchor_plane_occlusion
        self.anchor_ball_occlusion = anchor_ball_occlusion
        self.anchor_bottom_surface_occlusion = anchor_bottom_surface_occlusion
        
        self.distractor_anchor_aug = distractor_anchor_aug
        if self.distractor_anchor_aug:
            self.T_aug_list = []
            self.anchor_2_list = []
        
        self.num_demo = num_demo
        
        self.scale_point_clouds = scale_point_clouds
        self.scale_point_clouds_min = scale_point_clouds_min
        self.scale_point_clouds_max = scale_point_clouds_max
        
        self.gaussian_noise_mu = gaussian_noise_mu
        self.gaussian_noise_std = gaussian_noise_std

        self.demo_mod_k_range = demo_mod_k_range
        self.demo_mod_rot_var = demo_mod_rot_var
        self.demo_mod_trans_var = demo_mod_trans_var
        self.action_rot_sample_method = action_rot_sample_method
        self.anchor_rot_sample_method = anchor_rot_sample_method
        self.distractor_rot_sample_method = distractor_rot_sample_method
        
        self.downsample_type = downsample_type

        self.filenames = [
            self.dataset_root / f'{idx}_{self.cloud_type}_obj_points.npz' for idx in self.dataset_indices if idx not in self.bad_demo_id]
        if self.num_demo is not None:
            self.filenames = self.filenames[:self.num_demo]
            print(f'Using {self.num_demo} files: \n{self.filenames}')

        if self.overfit:
            self.get_fixed_transforms()
            
        if self.overfit_distactor_aug and self.distractor_anchor_aug:
            self.get_fixed_distractor_transforms()

    def get_fixed_distractor_transforms(self):
        print(f'num_overfit_transforms: {self.num_overfit_transforms}')
        print(f'num_demo: {self.num_demo}')
        for i in range(max(self.num_overfit_transforms, self.num_demo if self.num_demo is not None else 1)):
            points_action, points_anchor, _ = self.load_data(
                self.filenames[i],
                action_class=self.action_class,
                anchor_class=self.anchor_class,
            )
            if self.seed_overfit_transforms:
                torch.random.manual_seed(0)
            if self.distractor_anchor_aug:
                _, points_action, points_anchor1, points_anchor2, T_aug, debug = get_random_rack_demo(
                    None, 
                    points_action, 
                    points_anchor,
                    rot_sample_method=self.distractor_rot_sample_method
                )
                self.T_aug_list.append(T_aug)
                self.anchor_2_list.append(points_anchor2)

    def get_fixed_transforms(self):
        points_action, points_anchor, _ = self.load_data(
            self.filenames[0],
            action_class=self.action_class,
            anchor_class=self.anchor_class,
        )
        if self.overfit:
            if self.seed_overfit_transforms:
                torch.random.manual_seed(0)
            for i in range(self.num_overfit_transforms):
                a = random_se3(1, rot_var=self.rot_var,
                               trans_var=self.trans_var, device=points_action.device, rot_sample_method=self.action_rot_sample_method)
                b = random_se3(1, rot_var=self.rot_var,
                               trans_var=self.trans_var, device=points_anchor.device, rot_sample_method=self.anchor_rot_sample_method)
                self.T0_list.append(a)
                self.T1_list.append(b)
            print(f"\nOverfitting transform lists in PointCloudDataset:\n\t-TO: {[m.get_matrix() for m in self.T0_list]}\n\t-T1: {[m.get_matrix() for m in self.T1_list]}")
        return

    def get_existing_data_indices(self):
        import fnmatch
        num_files = len(fnmatch.filter(os.listdir(
            self.dataset_root), f'**_{self.cloud_type}_obj_points.npz'))
        file_indices = [int(fn.split('_')[0]) for fn in fnmatch.filter(
            os.listdir(self.dataset_root), f'**_{self.cloud_type}_obj_points.npz')]
        return file_indices

    def load_data(self, filename, action_class, anchor_class):
        point_data = np.load(filename, allow_pickle=True)
        points_raw_np = point_data['clouds']
        classes_raw_np = point_data['classes']
        if(self.min_num_cameras < 4):
            camera_idxs = np.concatenate([[0], np.cumsum((np.diff(classes_raw_np) == -2))])
            # if(not np.all(np.isin(np.arange(4), np.unique(camera_idxs)))):
            if(not np.all(np.isin(np.array([action_class, anchor_class]), np.unique(camera_idxs)))):
                print('\033[93m' + f'{filename} did not contain all classes in all cameras' +'\033[0m')
                return torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
            
            num_cameras = np.random.randint(low=self.min_num_cameras, high=self.max_num_cameras+1)
            sampled_camera_idxs = np.random.choice(4, num_cameras, replace=False)
            valid_idxs = np.isin(camera_idxs, sampled_camera_idxs)
            points_raw_np = points_raw_np[valid_idxs]
            classes_raw_np = classes_raw_np[valid_idxs]
            
        points_action_np = points_raw_np[classes_raw_np == action_class].copy()
        points_action_mean_np = points_action_np.mean(axis=0)
        points_action_np = points_action_np - points_action_mean_np

        points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()
        points_anchor_np = points_anchor_np - points_action_mean_np

        points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
        points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

        symmetric_cls = torch.Tensor([])

        return points_action, points_anchor, symmetric_cls

    def go_through_list(self):
        bad_demo_id = []
        filenames = [self.dataset_root /
                     f'{idx}_{self.cloud_type}_obj_points.npz' for idx in self.dataset_indices]
        for i in range(len(filenames)):
            filename = filenames[i]
            if i == 0:
                print(filename)
            if not os.path.exists(filename):
                bad_demo_id.append(i)
                continue
            points_action, points_anchor, _ = self.load_data(
                filename,
                action_class=self.action_class,
                anchor_class=self.anchor_class,
            )
            if(points_action.shape[1] < self.num_points) or (points_anchor.shape[1] < self.num_points):
                bad_demo_id.append(i)

        return bad_demo_id

    def project_to_xy(self, vector):
        """
        vector: num_points, 3
        """
        if len(vector.shape) > 1:
            vector[:, -1] = 0
        elif len(vector.shape) == 1:
            vector[-1] = 0
        return vector

    def get_sym_label(self, action_cloud, anchor_cloud, action_class, anchor_class, discrete=True):
        assert 0 in [
            action_class, anchor_class], "class 0 must be here somewhere as the manipulation object of interest"
        if action_class == 0:
            sym_breaking_class = action_class
            center_class = anchor_class
            points_sym = action_cloud[0]
            points_nonsym = anchor_cloud[0]
        elif anchor_class == 0:
            sym_breaking_class = anchor_class
            center_class = action_class
            points_sym = anchor_cloud[0]
            points_nonsym = action_cloud[0]

        non_sym_center = points_nonsym.mean(axis=0)
        sym_center = points_sym.mean(axis=0)
        sym2nonsym = (non_sym_center - sym_center)
        sym2nonsym = self.project_to_xy(sym2nonsym)

        sym_vec = points_sym - sym_center
        sym_vec = self.project_to_xy(sym_vec)
        if discrete:
            sym_cls = torch.sign(torch.matmul(sym_vec, sym2nonsym)
                                 ).unsqueeze(0)  # num_points, 1

        return sym_cls

    def downsample_pcd(self, points, type="fps"):
        if type == "fps":
            return sample_farthest_points(points, K=self.num_points, random_start_point=True)
        elif type == "random":
            random_idx = torch.randperm(points.shape[1])[:self.num_points]
            return points[:, random_idx], random_idx
        elif type.startswith("random_"):
            prob = float(type.split("_")[1])
            if np.random.random() < prob:
                return sample_farthest_points(points, K=self.num_points, random_start_point=True)
            else:
                random_idx = torch.randperm(points.shape[1])[:self.num_points]
                return points[:, random_idx], random_idx

    def get_data_index(self, index):
        if self.random_files:
            filename = self.filenames[torch.randint(
                len(self.filenames), [1])]
        else:
            filename = self.filenames[index % len(self.filenames)]
        points_action, points_anchor, symmetric_cls = self.load_data(
            filename,
            action_class=self.action_class,
            anchor_class=self.anchor_class,
        )

        # Get duplicate anchor point cloud for distractor
        if self.distractor_anchor_aug:
            if self.overfit_distactor_aug:
                points_anchor1 = points_anchor.clone()
                points_anchor2 = self.anchor_2_list[index % len(self.anchor_2_list)]
                T_aug_list = [self.T_aug_list[index % len(self.T_aug_list)]]
            else:
                _, points_action, points_anchor1, new_points_list, T_aug_list, debug = get_random_rack_demo(
                            None, 
                            points_action, 
                            points_anchor,
                            rot_sample_method=self.distractor_rot_sample_method,
                            num_racks_to_add=int(self.distractor_anchor_aug))
            points_anchor = torch.cat([points_anchor1] + new_points_list, axis=1)

        # Get transformations to apply to action and anchor point clouds 
        if self.overfit:
            transform_idx = torch.randint(
                self.num_overfit_transforms, (1,)).item()
            T0 = self.T0_list[transform_idx]
            T1 = self.T1_list[transform_idx]
        else:
            T0 = random_se3(1, rot_var=self.rot_var,
                            trans_var=self.trans_var, device=points_action.device, rot_sample_method=self.action_rot_sample_method)
            T1 = random_se3(1, rot_var=self.rot_var,
                            trans_var=self.trans_var, device=points_anchor.device, rot_sample_method=self.anchor_rot_sample_method)

        # Apply scaling to point clouds
        scale_factor = 1.0
        if self.scale_point_clouds:
            scale_factor = torch.rand(1) * (self.scale_point_clouds_max - self.scale_point_clouds_min) + self.scale_point_clouds_min

        # Apply occlusions to action point cloud
        start_points_action = points_action.clone()
        if(points_action.shape[1] > self.num_points):
            if np.random.random() < self.synthetic_occlusion:
                if self.action_ball_occlusion:
                    points_action = ball_occlusion(
                        points_action[0], radius=self.ball_radius).unsqueeze(0)
                if self.action_plane_occlusion:
                    points_action = plane_occlusion(
                        points_action[0], stand_off=self.plane_standoff).unsqueeze(0)
                if self.action_bottom_surface_occlusion:
                    points_action = bottom_surface_occlusion(
                        points_action[0], z_clipping_height=self.bottom_surface_z_clipping_height).unsqueeze(0)

            if(points_action.shape[1] > self.num_points):
                # Scale point cloud
                if self.scale_point_clouds:
                    points_action = points_action * scale_factor
                
                # TODO PUT THIS RANDOMNESS BACK
                points_action, action_ids = self.downsample_pcd(points_action, type=self.downsample_type)
                                                                   # K=self.num_points, random_start_point=True)
            elif(points_action.shape[1] < self.num_points):
                # Scale point cloud
                if self.scale_point_clouds:
                    start_points_action = start_points_action * scale_factor
                
                # Farthest point sampling
                points_action, action_ids = self.downsample_pcd(start_points_action, type=self.downsample_type)

            if(len(symmetric_cls) > 0):
                symmetric_cls = symmetric_cls[action_ids.view(-1)]
                
            if self.gaussian_noise_mu != 0 or self.gaussian_noise_std != 0:
                points_action = points_action + torch.normal(mean=self.gaussian_noise_mu, std=self.gaussian_noise_std, size=points_action.shape)
            
        elif(points_action.shape[1] < self.num_points):
            raise NotImplementedError(
                f'Action point cloud is smaller than cloud size ({points_action.shape[1]} < {self.num_points})')

        # Apply occlusions to anchor point cloud
        start_points_anchor = points_anchor.clone()
        if(points_anchor.shape[1] > self.num_points):
            if np.random.random() < self.synthetic_occlusion:
                if self.anchor_ball_occlusion:
                    points_anchor = ball_occlusion(
                        points_anchor[0], radius=self.ball_radius).unsqueeze(0)
                if self.anchor_plane_occlusion:
                    points_anchor = plane_occlusion(
                        points_anchor[0], stand_off=self.plane_standoff).unsqueeze(0)
                if self.anchor_bottom_surface_occlusion:
                    points_anchor = bottom_surface_occlusion(
                        points_anchor[0], z_clipping_height=self.bottom_surface_z_clipping_height).unsqueeze(0)
                    
            if(points_anchor.shape[1] > self.num_points):
                # Scale point cloud
                if self.scale_point_clouds:
                    points_anchor = points_anchor * scale_factor
                
                # TODO PUT THIS RANDOMNESS BACK
                points_anchor, _ = self.downsample_pcd(points_anchor, type=self.downsample_type)
                                                          # K=self.num_points, random_start_point=True)
            elif(points_anchor.shape[1] < self.num_points):
                # Scale point cloud
                if self.scale_point_clouds:
                    start_points_anchor = start_points_anchor * scale_factor
                
                # Farthest point sampling
                points_anchor, _ = self.downsample_pcd(start_points_anchor, type=self.downsample_type)
                    
            if self.gaussian_noise_mu != 0 or self.gaussian_noise_std != 0:
                points_anchor = points_anchor + torch.normal(mean=self.gaussian_noise_mu, std=self.gaussian_noise_std, size=points_anchor.shape)
                    
        elif(points_anchor.shape[1] < self.num_points):
            raise NotImplementedError(
                f'Anchor point cloud is smaller than cloud size ({points_anchor.shape[1]} < {self.num_points})')

        # Apply transformations to action.anchor points clouds from demo
        points_action_trans = T0.transform_points(points_action)
        points_anchor_trans = T1.transform_points(points_anchor)
            
        # Add symmetric class label
        if self.symmetric_class is not None:
            symmetric_cls = self.get_sym_label(
                action_cloud=points_action, anchor_cloud=points_anchor, action_class=self.action_class, anchor_class=self.anchor_class)  # num_points, 1
            symmetric_cls = symmetric_cls.unsqueeze(-1)
            if self.action_class == 0:

                points_action = torch.cat(
                    [points_action, symmetric_cls], axis=-1)

                points_anchor = torch.cat(
                    [points_anchor, torch.ones(symmetric_cls.shape)], axis=-1)
                points_action_trans = torch.cat(
                    [points_action_trans, symmetric_cls], axis=-1)
                points_anchor_trans = torch.cat(
                    [points_anchor_trans, torch.ones(symmetric_cls.shape)], axis=-1)

            elif self.anchor_class == 0:

                points_anchor = torch.cat(
                    [points_anchor, symmetric_cls], axis=-1)

                points_action = torch.cat(
                    [points_action, torch.ones(symmetric_cls.shape)], axis=-1)
                points_anchor_trans = torch.cat(
                    [points_anchor_trans, symmetric_cls], axis=-1)
                points_action_trans = torch.cat(
                    [points_action_trans, torch.ones(symmetric_cls.shape)], axis=-1)
                

        data = {
            'points_action': points_action.squeeze(0),
            'points_anchor': points_anchor.squeeze(0),
            'points_action_trans': points_action_trans.squeeze(0),
            'points_anchor_trans': points_anchor_trans.squeeze(0),
            'T0': T0.get_matrix().squeeze(0),
            'T1': T1.get_matrix().squeeze(0),
            'symmetric_cls': symmetric_cls,
            'mug_id': filename.name.split("_")[0],
        }

        if self.distractor_anchor_aug:
            data['T_aug_list'] = [T_aug.get_matrix().squeeze(0) for T_aug in T_aug_list]
            
            points_action_aug_trans_list = []
            for i, T_aug in enumerate(T_aug_list):
                points_action_aug_trans_cur = T_aug.transform_points(points_action[:, :, :3])
                points_action_aug_trans_cur = torch.cat([
                    points_action_aug_trans_cur,
                    points_action[:, :, 3:]
                ], axis=-1)
                points_action_aug_trans_list.append(points_action_aug_trans_cur)

            points_action_aug_trans = torch.cat(points_action_aug_trans_list, axis=1)
            data['points_action_aug_trans'] = points_action_aug_trans.squeeze(0)

        return data

    def __getitem__(self, index):
        return self.get_data_index(index)

    def __len__(self):
        return self.dataset_size
