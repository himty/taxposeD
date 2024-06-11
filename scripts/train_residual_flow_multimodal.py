# from equivariant_pose_graph.dataset.rpdiff_data_module import RpDiffDataModule
from equivariant_pose_graph.training.flow_equivariance_training_module_nocentering_multimodal import EquivarianceTrainingModule, EquivarianceTrainingModule_WithPZCondX
from equivariant_pose_graph.dataset.point_cloud_data_module import MultiviewDataModule
from equivariant_pose_graph.models.transformer_flow import ResidualFlow_DiffEmbTransformer, FlowDecoder
from equivariant_pose_graph.models.multimodal_transformer_flow import Multimodal_ResidualFlow_DiffEmbTransformer, Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX
import os
import torch
import torch.nn as nn

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import pytorch_lightning as pl

from equivariant_pose_graph.dataset.point_cloud_data_module import MultiviewDataModule
import hydra
from equivariant_pose_graph.utils.callbacks import SaverCallbackModel, SaverCallbackEmbnnActionAnchorMultimodal
from pytorch_lightning.loggers import WandbLogger


def setup_main(cfg):
    pl.seed_everything(cfg.seed)
    if cfg.resume_id is None:
        logger = WandbLogger(project=cfg.experiment, group=cfg.wandb_group)
    else:
        logger = WandbLogger(project=cfg.experiment, group=cfg.wandb_group, id=cfg.resume_id, resume="must")
    logger.log_hyperparams(cfg)
    logger.log_hyperparams({'working_dir': os.getcwd()})
    
    trainer = pl.Trainer(logger=logger,
                         gpus=1,
                         reload_dataloaders_every_n_epochs=1,
                         callbacks=[SaverCallbackModel(save_freq=cfg.ckpt_save_freq)],#, SaverCallbackEmbnnActionAnchorMultimodal()],
                         max_steps=cfg.max_steps,
                         gradient_clip_val=cfg.gradient_clipping, # 0 is no clipping
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    dm = MultiviewDataModule(
        dataset_root=cfg.dataset_root,
        test_dataset_root=cfg.test_dataset_root,
        dataset_index=cfg.dataset_index,
        action_class=cfg.action_class,
        anchor_class=cfg.anchor_class,
        dataset_size=cfg.dataset_size,
        rotation_variance=np.pi/180 * cfg.rotation_variance,
        translation_variance=cfg.translation_variance,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cloud_type=cfg.cloud_type,
        num_points=cfg.num_points,
        overfit=cfg.overfit,
        overfit_distractor_aug=cfg.overfit_distractor_aug,
        num_overfit_transforms=cfg.num_overfit_transforms,
        seed_overfit_transforms=cfg.seed_overfit_transforms,
        num_demo=cfg.num_demo,
        synthetic_occlusion=cfg.synthetic_occlusion,
        ball_radius=cfg.ball_radius,
        plane_standoff=cfg.plane_standoff,
        bottom_surface_z_clipping_height=cfg.bottom_surface_z_clipping_height,
        scale_point_clouds=cfg.scale_point_clouds,
        scale_point_clouds_min=cfg.scale_point_clouds_min,
        scale_point_clouds_max=cfg.scale_point_clouds_max,
        distractor_anchor_aug=cfg.distractor_anchor_aug,
        demo_mod_k_range=[cfg.demo_mod_k_range_min, cfg.demo_mod_k_range_max],
        demo_mod_rot_var=cfg.demo_mod_rot_var * np.pi/180,
        demo_mod_trans_var=cfg.demo_mod_trans_var,
        action_rot_sample_method=cfg.action_rot_sample_method,
        anchor_rot_sample_method=cfg.anchor_rot_sample_method,
        distractor_rot_sample_method=cfg.distractor_rot_sample_method,
        min_num_cameras=cfg.min_num_cameras,
        max_num_cameras=cfg.max_num_cameras,
        use_consistent_validation_set=cfg.use_consistent_validation_set,
        use_all_validation_sets=cfg.use_all_validation_sets,
        conval_rotation_variance=np.pi/180 * cfg.conval_rotation_variance,
        conval_translation_variance=cfg.conval_translation_variance,
        conval_synthetic_occlusion=cfg.conval_synthetic_occlusion,
        conval_scale_point_clouds=cfg.conval_scale_point_clouds,
        conval_action_rot_sample_method=cfg.conval_action_rot_sample_method,
        conval_anchor_rot_sample_method=cfg.conval_anchor_rot_sample_method,
        conval_distractor_rot_sample_method=cfg.conval_distractor_rot_sample_method,
        conval_min_num_cameras=cfg.conval_min_num_cameras,
        conval_max_num_cameras=cfg.conval_max_num_cameras,
        conval_downsample_type=cfg.conval_downsample_type,
        conval_gaussian_noise_mu=cfg.conval_gaussian_noise_mu,
        conval_gaussian_noise_std=cfg.conval_gaussian_noise_std,
        action_plane_occlusion=cfg.action_plane_occlusion,
        action_ball_occlusion=cfg.action_ball_occlusion,
        action_bottom_surface_occlusion=cfg.action_bottom_surface_occlusion,
        anchor_plane_occlusion=cfg.anchor_plane_occlusion,
        anchor_ball_occlusion=cfg.anchor_ball_occlusion,
        anchor_bottom_surface_occlusion=cfg.anchor_bottom_surface_occlusion,
        downsample_type=cfg.downsample_type,
        gaussian_noise_mu=cfg.gaussian_noise_mu,
        gaussian_noise_std=cfg.gaussian_noise_std,
    )

    dm.setup()
    
    TP_input_dims = Multimodal_ResidualFlow_DiffEmbTransformer.TP_INPUT_DIMS[cfg.conditioning]

    inner_network = ResidualFlow_DiffEmbTransformer(
        emb_dims=cfg.emb_dims,
        input_dims=TP_input_dims,
        emb_nn=cfg.emb_nn,
        return_flow_component=cfg.return_flow_component,
        center_feature=cfg.center_feature,
        pred_weight=cfg.pred_weight,
        freeze_embnn=cfg.freeze_embnn,
        conditioning_size=cfg.latent_z_linear_size if cfg.conditioning in ["latent_z_linear_internalcond"] else 0,
        conditioning_type=cfg.taxpose_conditioning_type,
    )

    network = Multimodal_ResidualFlow_DiffEmbTransformer(
        residualflow_diffembtransformer=inner_network,
        gumbel_temp=cfg.gumbel_temp,
        freeze_residual_flow=cfg.freeze_residual_flow,
        freeze_z_embnn=cfg.freeze_z_embnn,
        add_smooth_factor=cfg.add_smooth_factor,
        conditioning=cfg.conditioning,
        latent_z_linear_size=cfg.latent_z_linear_size,
        taxpose_centering=cfg.taxpose_centering,
        pzY_input_dims=cfg.pzY_input_dims,
        latent_z_cond_logvar_limit=cfg.latent_z_cond_logvar_limit,
    )

    model = EquivarianceTrainingModule(
        network,
        lr=cfg.lr,
        image_log_period=cfg.image_logging_period,
        flow_supervision=cfg.flow_supervision,
        action_weight=cfg.action_weight,
        anchor_weight=cfg.anchor_weight,
        displace_weight=cfg.displace_weight,
        consistency_weight=cfg.consistency_weight,
        smoothness_weight=cfg.smoothness_weight,
        rotation_weight=cfg.rotation_weight,
        weight_normalize=cfg.weight_normalize,
        softmax_temperature=cfg.softmax_temperature,
        vae_reg_loss_weight=cfg.vae_reg_loss_weight,
        sigmoid_on=cfg.sigmoid_on,
        min_err_across_racks_debug=cfg.min_err_across_racks_debug,
        error_mode_2rack=cfg.error_mode_2rack,
        return_flow_component=cfg.return_flow_component,
        plot_encoder_distribution=cfg.plot_encoder_distribution,
    )

    if cfg.init_cond_x:
        network_cond_x = Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(
            residualflow_embnn=network,
            pzX_transformer_embnn_dims=cfg.pzX_transformer_embnn_dims,
            pzX_transformer_emb_dims=cfg.pzX_transformer_emb_dims,
            pzX_input_dims=cfg.pzX_input_dims,
        )

        model_cond_x = EquivarianceTrainingModule_WithPZCondX(
            network_cond_x,
            model,
            goal_emb_cond_x_loss_weight=cfg.goal_emb_cond_x_loss_weight,
            freeze_residual_flow=cfg.freeze_residual_flow,
            freeze_z_embnn=cfg.freeze_z_embnn,
            freeze_embnn=cfg.freeze_embnn,
            plot_encoder_distribution=cfg.plot_encoder_distribution,
            goal_emb_cond_x_loss_type=cfg.goal_emb_cond_x_loss_type,
            overwrite_loss=cfg.pzX_overwrite_loss,
        )

        model_cond_x.cuda()
        model_cond_x.train()        
    else:
        model.cuda()
        model.train()
     
    
    if(cfg.checkpoint_file is not None):
        print("loaded checkpoint from")
        print(cfg.checkpoint_file)
        if not cfg.load_cond_x:
            model.load_state_dict(torch.load(cfg.checkpoint_file)['state_dict'])
            
            if cfg.init_cond_x and cfg. load_pretraining_for_conditioning:
                if cfg.checkpoint_file_action is not None:
                    model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.conv5 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
                    model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.bn5 = nn.BatchNorm2d(512)
                    model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.load_state_dict(
                        torch.load(cfg.checkpoint_file_action)['embnn_state_dict'])
                    model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.conv5 = nn.Conv2d(512, TP_input_dims-3, kernel_size=1, bias=False)
                    model_cond_x.model_with_cond_x.p_z_cond_x_embnn_action.bn5 = nn.BatchNorm2d(TP_input_dims-3)
                    print("----Action Pretraining for p(z|X) Loaded!----")
                if cfg.checkpoint_file_anchor is not None:
                        print("--Not loading p(z|X) for anchor for now--")
        else:
            model_cond_x.load_state_dict(torch.load(cfg.checkpoint_file)['state_dict'])

    else:
        if cfg.checkpoint_file_action is not None:
            if cfg.load_pretraining_for_taxpose:
                model.model.tax_pose.emb_nn_action.conv1 = nn.Conv2d(3*2, 64, kernel_size=1, bias=False)
                model.model.tax_pose.emb_nn_action.load_state_dict(
                    torch.load(cfg.checkpoint_file_action)['embnn_state_dict'])
                model.model.tax_pose.emb_nn_action.conv1 = nn.Conv2d(TP_input_dims*2, 64, kernel_size=1, bias=False)
                print(
                '-----------------------Pretrained EmbNN Action Model Loaded!-----------------------')
            if cfg.load_pretraining_for_conditioning:
                if not cfg.init_cond_x:
                    print("---Not Loading p(z|Y) Pretraining For Now---")
                    pass
            
        if cfg.checkpoint_file_anchor is not None:
            if cfg.load_pretraining_for_taxpose:
                model.model.tax_pose.emb_nn_anchor.conv1 = nn.Conv2d(3*2, 64, kernel_size=1, bias=False)
                model.model.tax_pose.emb_nn_anchor.load_state_dict(
                    torch.load(cfg.checkpoint_file_anchor)['embnn_state_dict'])
                model.model.tax_pose.emb_nn_anchor.conv1 = nn.Conv2d(TP_input_dims*2, 64, kernel_size=1, bias=False)
                print(
                '-----------------------Pretrained EmbNN Anchor Model Loaded!-----------------------')
            if cfg.load_pretraining_for_conditioning:
                if not cfg.init_cond_x:
                    print("---Not Loading p(z|Y) Pretraining For Now---")
                    pass

    if cfg.init_cond_x:
        return trainer, model_cond_x, dm
    else:
        return trainer, model, dm
    
@hydra.main(config_path="../configs", config_name="train_mug_residual")  
def main(cfg):
    trainer, model, dm = setup_main(cfg)
    trainer.fit(model, dm)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
