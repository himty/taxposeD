from torch.utils.data import DataLoader
import numpy as np

import pytorch_lightning as pl
from equivariant_pose_graph.dataset.point_cloud_dataset import PointCloudDataset

class MultiviewDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_root,
                 test_dataset_root,
                 dataset_index=10,
                 action_class=0,
                 anchor_class=1,
                 dataset_size=1000,
                 rotation_variance=np.pi/180 * 5,
                 translation_variance=0.1,
                 batch_size=8,
                 num_workers=8,
                 cloud_type='final',
                 symmetric_class=None,
                 num_points=1024,
                 overfit=False,
                 overfit_distractor_aug=False,
                 num_overfit_transforms=3,
                 seed_overfit_transforms=False,
                 no_transform_applied=False,
                 init_distribution_tranform_file='',
                 synthetic_occlusion=False,
                 ball_radius=None,
                 plane_standoff=None,
                 bottom_surface_z_clipping_height=0.1,
                 scale_point_clouds=False,
                 scale_point_clouds_min=0.5,
                 scale_point_clouds_max=2.0,
                 distractor_anchor_aug=False,
                 num_demo=12,
                 demo_mod_k_range=[2, 2],
                 demo_mod_rot_var=np.pi/180 * 360,
                 demo_mod_trans_var=0.15,
                 action_rot_sample_method="axis_angle",
                 anchor_rot_sample_method="axis_angle",
                 distractor_rot_sample_method="axis_angle",
                 min_num_cameras=4,
                 max_num_cameras=4,
                 use_all_validation_sets=False,
                 use_consistent_validation_set=False,
                 conval_rotation_variance=180,
                 conval_translation_variance=0.5,
                 conval_synthetic_occlusion=False,
                 conval_scale_point_clouds=False,
                 conval_action_rot_sample_method="quat_uniform",
                 conval_anchor_rot_sample_method="random_flat_upright",
                 conval_distractor_rot_sample_method="random_flat_upright",
                 conval_min_num_cameras=4,
                 conval_max_num_cameras=4,
                 conval_downsample_type="fps",
                 conval_gaussian_noise_mu=0.0,
                 conval_gaussian_noise_std=0.001,
                 action_plane_occlusion=True,
                 action_ball_occlusion=True,
                 action_bottom_surface_occlusion=True,
                 anchor_plane_occlusion=False,
                 anchor_ball_occlusion=False,
                 anchor_bottom_surface_occlusion=False,
                 downsample_type="fps",
                 gaussian_noise_mu=0.0,
                 gaussian_noise_std=0.001,
                 ):
        super().__init__()
        self.dataset_root = dataset_root
        self.test_dataset_root = test_dataset_root
        if isinstance(dataset_index, list):
            self.dataset_index = dataset_index
        elif dataset_index == None:
            self.dataset_index = None
        self.dataset_index = dataset_index
        self.no_transform_applied = no_transform_applied

        self.action_class = action_class
        self.anchor_class = anchor_class
        self.dataset_size = dataset_size
        self.rotation_variance = rotation_variance
        self.translation_variance = translation_variance
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cloud_type = cloud_type
        self.symmetric_class = symmetric_class
        self.num_points = num_points
        self.overfit = overfit
        self.overfit_distractor_aug = overfit_distractor_aug
        self.num_overfit_transforms = num_overfit_transforms
        self.seed_overfit_transforms = seed_overfit_transforms
        self.index_list = []
        self.init_distribution_tranform_file = init_distribution_tranform_file
        self.synthetic_occlusion = synthetic_occlusion
        self.ball_radius = ball_radius
        self.plane_standoff = plane_standoff
        self.bottom_surface_z_clipping_height = bottom_surface_z_clipping_height
        self.scale_point_clouds = scale_point_clouds
        self.scale_point_clouds_min = scale_point_clouds_min
        self.scale_point_clouds_max = scale_point_clouds_max
        self.distractor_anchor_aug = distractor_anchor_aug
        self.num_demo = num_demo
        self.min_num_cameras = min_num_cameras
        self.max_num_cameras = max_num_cameras
        
        self.action_plane_occlusion = action_plane_occlusion
        self.action_ball_occlusion = action_ball_occlusion
        self.action_bottom_surface_occlusion = action_bottom_surface_occlusion
        self.anchor_plane_occlusion = anchor_plane_occlusion
        self.anchor_ball_occlusion = anchor_ball_occlusion
        self.anchor_bottom_surface_occlusion = anchor_bottom_surface_occlusion

        self.demo_mod_k_range = demo_mod_k_range
        self.demo_mod_rot_var = demo_mod_rot_var
        self.demo_mod_trans_var = demo_mod_trans_var
        self.action_rot_sample_method = action_rot_sample_method
        self.anchor_rot_sample_method = anchor_rot_sample_method
        self.distractor_rot_sample_method = distractor_rot_sample_method
        
        self.use_all_validation_sets = use_all_validation_sets
        self.use_consistent_validation_set = use_consistent_validation_set
        self.conval_rotation_variance = conval_rotation_variance
        self.conval_translation_variance = conval_translation_variance
        self.conval_synthetic_occlusion = conval_synthetic_occlusion
        self.conval_scale_point_clouds = conval_scale_point_clouds
        self.conval_action_rot_sample_method = conval_action_rot_sample_method
        self.conval_anchor_rot_sample_method = conval_anchor_rot_sample_method
        self.conval_distractor_rot_sample_method = conval_distractor_rot_sample_method
        self.conval_min_num_cameras = conval_min_num_cameras
        self.conval_max_num_cameras = conval_max_num_cameras
        self.conval_downsample_type = conval_downsample_type
        self.conval_gaussian_noise_mu = conval_gaussian_noise_mu
        self.conval_gaussian_noise_std = conval_gaussian_noise_std
        
        self.downsample_type = downsample_type
        
        self.gaussian_noise_mu = gaussian_noise_mu
        self.gaussian_noise_std = gaussian_noise_std
        
        self.val_datasets = []

    def pass_loss(self, loss):
        self.loss = loss.to(self.device)

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        pass

    def setup(self, stage=None):
        '''called one each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            print("TRAIN Dataset")
            print(self.dataset_root)
            self.train_dataset = PointCloudDataset(
                dataset_root=self.dataset_root,
                dataset_indices=self.dataset_index,  # [self.dataset_index],
                action_class=self.action_class,
                anchor_class=self.anchor_class,
                dataset_size=self.dataset_size,
                rotation_variance=self.rotation_variance,
                translation_variance=self.translation_variance,
                cloud_type=self.cloud_type,
                symmetric_class=self.symmetric_class,
                num_points=self.num_points,
                overfit=self.overfit,
                overfit_distractor_aug=self.overfit_distractor_aug,
                num_overfit_transforms=self.num_overfit_transforms,
                seed_overfit_transforms=self.seed_overfit_transforms,
                synthetic_occlusion=self.synthetic_occlusion,
                ball_radius=self.ball_radius,
                plane_standoff=self.plane_standoff,
                bottom_surface_z_clipping_height=self.bottom_surface_z_clipping_height,
                scale_point_clouds=self.scale_point_clouds,
                scale_point_clouds_min=self.scale_point_clouds_min,
                scale_point_clouds_max=self.scale_point_clouds_max,
                distractor_anchor_aug=self.distractor_anchor_aug,
                num_demo=self.num_demo,
                demo_mod_k_range=self.demo_mod_k_range,
                demo_mod_rot_var=self.demo_mod_rot_var,
                demo_mod_trans_var=self.demo_mod_trans_var,
                action_rot_sample_method=self.action_rot_sample_method,
                anchor_rot_sample_method=self.anchor_rot_sample_method,
                distractor_rot_sample_method=self.distractor_rot_sample_method,
                random_files=True,
                min_num_cameras=self.min_num_cameras,
                max_num_cameras=self.max_num_cameras,
                action_plane_occlusion=self.action_plane_occlusion,
                action_ball_occlusion=self.action_ball_occlusion,
                action_bottom_surface_occlusion=self.action_bottom_surface_occlusion,
                anchor_plane_occlusion=self.anchor_plane_occlusion,
                anchor_ball_occlusion=self.anchor_ball_occlusion,
                anchor_bottom_surface_occlusion=self.anchor_bottom_surface_occlusion,
                downsample_type=self.downsample_type,
                gaussian_noise_mu=self.gaussian_noise_mu,
                gaussian_noise_std=self.gaussian_noise_std
            )

        if stage == 'val' or stage is None:
            val_params = []
            if self.use_all_validation_sets:
                val_params.append({
                    "rotation_variance": np.pi/180 * 180,
                    "translation_variance": 0.5,
                    "synthetic_occlusion": 0,
                    "scale_point_clouds": 0,
                    "action_rot_sample_method": "axis_angle",
                    "anchor_rot_sample_method": "axis_angle_uniform_z",
                    "distractor_rot_sample_method": "axis_angle_uniform_z",
                    "min_num_cameras": 4,
                    "max_num_cameras": 4,
                    "downsample_type": "fps",
                    "gaussian_noise_mu": 0.0,
                    "gaussian_noise_std": 0.001
                })
                val_params.append({
                    "rotation_variance": np.pi/180 * 180,
                    "translation_variance": 0.5,
                    "synthetic_occlusion": 0,
                    "scale_point_clouds": 0,
                    "action_rot_sample_method": "axis_angle_uniform_z",
                    "anchor_rot_sample_method": "axis_angle_uniform_z",
                    "distractor_rot_sample_method": "axis_angle_uniform_z",
                    "min_num_cameras": 4,
                    "max_num_cameras": 4,
                    "downsample_type": "fps",
                    "gaussian_noise_mu": 0.0,
                    "gaussian_noise_std": 0.001
                })
                val_params.append({
                    "rotation_variance": np.pi/180 * 180,
                    "translation_variance": 0.5,
                    "synthetic_occlusion": 0,
                    "scale_point_clouds": 0,
                    "action_rot_sample_method": "quat_uniform",
                    "anchor_rot_sample_method": "axis_angle_uniform_z",
                    "distractor_rot_sample_method": "axis_angle_uniform_z",
                    "min_num_cameras": 4,
                    "max_num_cameras": 4,
                    "downsample_type": "fps",
                    "gaussian_noise_mu": 0.0,
                    "gaussian_noise_std": 0.001
                })
            else:
                if self.use_consistent_validation_set:
                    val_params.append({
                        "rotation_variance": self.conval_rotation_variance,
                        "translation_variance": self.conval_translation_variance,
                        "synthetic_occlusion": self.conval_synthetic_occlusion,
                        "scale_point_clouds": self.conval_scale_point_clouds,
                        "action_rot_sample_method": self.conval_action_rot_sample_method,
                        "anchor_rot_sample_method": self.conval_anchor_rot_sample_method,
                        "distractor_rot_sample_method": self.conval_distractor_rot_sample_method,
                        "min_num_cameras": self.conval_min_num_cameras,
                        "max_num_cameras": self.conval_max_num_cameras,
                        "downsample_type": self.conval_downsample_type,
                        "gaussian_noise_mu": self.conval_gaussian_noise_mu,
                        "gaussian_noise_std": self.conval_gaussian_noise_std
                    })
                
                else:
                    val_params.append({
                        "rotation_variance": self.rotation_variance,
                        "translation_variance": self.translation_variance,
                        "synthetic_occlusion": self.synthetic_occlusion,
                        "scale_point_clouds": self.scale_point_clouds,
                        "action_rot_sample_method": self.action_rot_sample_method,
                        "anchor_rot_sample_method": self.anchor_rot_sample_method,
                        "distractor_rot_sample_method": self.distractor_rot_sample_method,
                        "min_num_cameras": self.min_num_cameras,
                        "max_num_cameras": self.max_num_cameras,
                        "downsample_type": self.downsample_type,
                        "gaussian_noise_mu": self.gaussian_noise_mu,
                        "gaussian_noise_std": self.gaussian_noise_std
                    })


            assert len(val_params) > 0, "No validation parameters specified"
            for val_idx, val_param in enumerate(val_params):
                print(f"VAL {val_idx} Dataset")
                self.val_datasets.append(
                    PointCloudDataset(
                        dataset_root=self.test_dataset_root,
                        dataset_indices=self.dataset_index,
                        action_class=self.action_class,
                        anchor_class=self.anchor_class,
                        dataset_size=self.dataset_size,
                        rotation_variance=val_param["rotation_variance"],
                        translation_variance=val_param["translation_variance"],
                        cloud_type=self.cloud_type,
                        symmetric_class=self.symmetric_class,
                        num_points=self.num_points,
                        overfit=self.overfit,
                        overfit_distractor_aug=self.overfit_distractor_aug,
                        num_overfit_transforms=self.num_overfit_transforms,
                        seed_overfit_transforms=self.seed_overfit_transforms,
                        synthetic_occlusion=val_param["synthetic_occlusion"],
                        ball_radius=self.ball_radius,
                        plane_standoff=self.plane_standoff,
                        bottom_surface_z_clipping_height=self.bottom_surface_z_clipping_height,
                        scale_point_clouds=val_param["scale_point_clouds"],
                        scale_point_clouds_min=self.scale_point_clouds_min,
                        scale_point_clouds_max=self.scale_point_clouds_max,
                        distractor_anchor_aug=self.distractor_anchor_aug,
                        num_demo=None,
                        demo_mod_k_range=self.demo_mod_k_range,
                        demo_mod_rot_var=self.demo_mod_rot_var,
                        demo_mod_trans_var=self.demo_mod_trans_var,
                        action_rot_sample_method=val_param["action_rot_sample_method"],
                        anchor_rot_sample_method=val_param["anchor_rot_sample_method"],
                        distractor_rot_sample_method=val_param["distractor_rot_sample_method"],
                        random_files=False,
                        min_num_cameras=val_param["min_num_cameras"],
                        max_num_cameras=val_param["max_num_cameras"],
                        action_plane_occlusion=self.action_plane_occlusion,
                        action_ball_occlusion=self.action_ball_occlusion,
                        action_bottom_surface_occlusion=self.action_bottom_surface_occlusion,
                        anchor_plane_occlusion=self.anchor_plane_occlusion,
                        anchor_ball_occlusion=self.anchor_ball_occlusion,
                        anchor_bottom_surface_occlusion=self.anchor_bottom_surface_occlusion,
                        downsample_type=val_param["downsample_type"],
                        gaussian_noise_mu=val_param["gaussian_noise_mu"],
                        gaussian_noise_std=val_param["gaussian_noise_std"]
                    )
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(val_dataset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers)
            )
        return val_dataloaders
    