from equivariant_pose_graph.models.transformer_flow import ResidualFlow_DiffEmbTransformer
from equivariant_pose_graph.models.multimodal_transformer_flow import Multimodal_ResidualFlow_DiffEmbTransformer, Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX
from equivariant_pose_graph.training.flow_equivariance_training_module_nocentering_multimodal import EquivarianceTrainingModule, EquivarianceTrainingModule_WithPZCondX
import torch

def load_model(place_checkpoint_file, has_pzX, conditioning="pos_delta_l2norm", return_flow_component=None, cfg=None):
    if cfg is not None:
        print("Loading model with architecture specified in configs")
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
            sample=cfg.mlat_sample,
            mlat_nkps=cfg.mlat_nkps,
            pred_mlat_weight=cfg.pred_mlat_weight,
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

        place_nocond_model = EquivarianceTrainingModule(
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
        
        if has_pzX:
            network_cond_x = Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(
                residualflow_embnn=network,
                pzX_transformer_embnn_dims=cfg.pzX_transformer_embnn_dims,
                pzX_transformer_emb_dims=cfg.pzX_transformer_emb_dims,
                pzX_input_dims=cfg.pzX_input_dims,
            )

            place_model = EquivarianceTrainingModule_WithPZCondX(
                network_cond_x,
                place_nocond_model,
                goal_emb_cond_x_loss_weight=cfg.goal_emb_cond_x_loss_weight,
                freeze_residual_flow=cfg.freeze_residual_flow,
                freeze_z_embnn=cfg.freeze_z_embnn,
                freeze_embnn=cfg.freeze_embnn,
                plot_encoder_distribution=cfg.plot_encoder_distribution,
                goal_emb_cond_x_loss_type=cfg.goal_emb_cond_x_loss_type,
                overwrite_loss=cfg.pzX_overwrite_loss,
            )
        else:
            place_model = place_nocond_model
    else:
        print("WARNING: No configs for the model are specified. Using default configs to load the model")
        inner_network = ResidualFlow_DiffEmbTransformer(
                    emb_nn='dgcnn', return_flow_component=return_flow_component, center_feature=True,
                    input_dims=4)
        place_nocond_network = Multimodal_ResidualFlow_DiffEmbTransformer(
                            inner_network, gumbel_temp=1, conditioning=conditioning)
        place_nocond_model = EquivarianceTrainingModule(
            place_nocond_network,
            lr=1e-4,
            image_log_period=100,
            weight_normalize='softmax', #'l1',
            softmax_temperature=1
        )
        
        if has_pzX:
            place_network = Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(
                            place_nocond_network, sample_z=False)
            place_model = EquivarianceTrainingModule_WithPZCondX(
                place_network,
                place_nocond_model,
            )
        else:
            place_model = place_nocond_model

    place_model.cuda()

    if place_checkpoint_file is not None:
        place_model.load_state_dict(torch.load(place_checkpoint_file)['state_dict'])
    else:
        print("WARNING: NO CHECKPOINT FILE SPECIFIED. THIS IS A DEBUG RUN WITH RANDOM WEIGHTS")
    return place_model

def load_merged_model(place_checkpoint_file_pzY, place_checkpoint_file_pzX, conditioning="pos_delta_l2norm", return_flow_component=None, cfg=None):
    if place_checkpoint_file_pzY is not None:
        pzY_model = load_model(place_checkpoint_file_pzY, has_pzX=False, conditioning=conditioning, return_flow_component=return_flow_component, cfg=cfg)
    
        if place_checkpoint_file_pzX is not None:
            pzX_model = load_model(place_checkpoint_file_pzX, has_pzX=True, conditioning=conditioning, return_flow_component=return_flow_component, cfg=cfg)
            pzX_model.model.tax_pose = pzY_model.model.tax_pose
            return pzX_model
        else:
            return pzY_model
    else:
        if place_checkpoint_file_pzX is not None:
            pzX_model = load_model(place_checkpoint_file_pzX, has_pzX=True, conditioning=conditioning, return_flow_component=return_flow_component, cfg=cfg)
            return pzX_model
        else:
            raise ValueError("No checkpoint file specified for either pzY or pzX. Cannot load a model")
