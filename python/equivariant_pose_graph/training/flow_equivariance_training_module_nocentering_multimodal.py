import numpy as np
import torch
from torch import nn

import plotly
import plotly.graph_objects as go
from pytorch3d.transforms import Transform3d
from torchvision.transforms import ToTensor
from equivariant_pose_graph.training.point_cloud_training_module import PointCloudTrainingModule
from equivariant_pose_graph.utils.se3 import dualflow2pose, flow2pose, get_translation, get_degree_angle, dense_flow_loss, pure_translation_se3
from equivariant_pose_graph.utils.color_utils import get_color, color_gradient
from equivariant_pose_graph.utils.error_metrics import get_2rack_errors, get_all_sample_errors
from equivariant_pose_graph.utils.loss_utils import js_div
from torch.distributions.normal import Normal

import torch.nn.functional as F

import wandb

mse_criterion = nn.MSELoss(reduction='sum')
to_tensor = ToTensor()


class EquivarianceTrainingModule(PointCloudTrainingModule):
    def __init__(self,
                 model=None,
                 lr=1e-3,
                 image_log_period=500,
                 flow_supervision="both",
                 action_weight=1,
                 anchor_weight=1,
                 displace_weight=1,
                 smoothness_weight=0.1,
                 consistency_weight=1,
                 vae_reg_loss_weight=0.01,
                 rotation_weight=0,
                 chamfer_weight=10000,
                 return_flow_component=False,
                 weight_normalize='l1',
                 sigmoid_on=False,
                 softmax_temperature=None,
                 min_err_across_racks_debug=False,
                 error_mode_2rack="batch_min_rack",
                 plot_encoder_distribution=False,
    ):
        super().__init__(model=model, lr=lr,
                         image_log_period=image_log_period,)
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period
        
        self.flow_supervision = flow_supervision
        self.action_weight = action_weight
        self.anchor_weight = anchor_weight
        self.displace_weight = displace_weight
        self.smoothness_weight = smoothness_weight
        self.consistency_weight = consistency_weight
        self.chamfer_weight = chamfer_weight
        self.rotation_weight = rotation_weight
        
        # This is only used when the internal model uses self.model.conditioning == "latent_z_internalcond" or "uniform_prior_pos_delta_l2norm"
        self.vae_reg_loss_weight = vae_reg_loss_weight
        self.display_action = True
        self.display_anchor = True
        self.weight_normalize = weight_normalize
        self.sigmoid_on = sigmoid_on
        self.softmax_temperature = softmax_temperature
        self.min_err_across_racks_debug = min_err_across_racks_debug
        self.error_mode_2rack = error_mode_2rack
        self.n_samples = 1
        self.plot_encoder_distribution = plot_encoder_distribution
        
        if self.weight_normalize == 'l1':
            assert (self.sigmoid_on), "l1 weight normalization need sigmoid on"

        self.get_sample_errors = True
        

    def action_centered(self, points_action, points_anchor):
        """
        @param points_action, (1,num_points,3)
        @param points_anchor, (1,num_points,3)
        """
        points_action_mean = points_action.clone().mean(axis=1)
        points_action_mean_centered = points_action-points_action_mean
        points_anchor_mean_centered = points_anchor-points_action_mean

        return points_action_mean_centered, points_anchor_mean_centered, points_action_mean

    def get_transform(self, points_trans_action, points_trans_anchor, points_action=None, points_anchor=None, mode="forward", sampling_method="gumbel", n_samples=1):
        model_outputs = self.model(
            points_trans_action, 
            points_trans_anchor, 
            points_action, 
            points_anchor, 
            mode=mode, 
            sampling_method=sampling_method, 
            n_samples=n_samples
        )

        ans_dicts = []
        for i in range(n_samples):
            model_output = model_outputs[i]

            points_trans_action = points_trans_action[:, :, :3]
            points_trans_anchor = points_trans_anchor[:, :, :3]
            
            ans_dict = self.predict(model_output=model_output,
                                    points_trans_action=points_trans_action, points_trans_anchor=points_trans_anchor)
            
            ans_dict['flow_components'] = model_output
            ans_dicts.append(ans_dict)
        return ans_dicts

    def predict(self, model_output, points_trans_action, points_trans_anchor):
        
        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_output:
            ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
            sampled_points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )
        else:
            sampled_points_trans_action = points_trans_action

        if "sampled_ixs_anchor" in model_output:
            ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
            sampled_points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )
        else:
            sampled_points_trans_anchor = points_trans_anchor

        # Get predicted transform
        if self.flow_supervision == "both":
            # Extract the predicted flow and weights
            x_action = model_output['flow_action']
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            x_anchor = model_output['flow_anchor']
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            
            pred_T_action = dualflow2pose(
                xyz_src=sampled_points_trans_action, 
                xyz_tgt=sampled_points_trans_anchor,
                flow_src=pred_flow_action, 
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action, 
                weights_tgt=pred_w_anchor,
                return_transform3d=True, 
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature
            )
        elif self.flow_supervision == "action2anchor":
            # Extract the predicted flow and weights
            x_action = model_output['flow_action']
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            pred_T_action = flow2pose(
                xyz=sampled_points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature
            )
        elif self.flow_supervision == "anchor2action":
            # Extract the predicted flow and weights
            x_anchor = model_output['flow_anchor']
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            
            pred_T_anchor = flow2pose(
                xyz=sampled_points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature
            )
            pred_T_action = pred_T_anchor.inverse()
        else:
            raise ValueError(f"ERROR: Invalid flow supervision type: {self.flow_supervision}")
            
        pred_points_action = pred_T_action.transform_points(points_trans_action)

        return {"pred_T_action": pred_T_action,
                "pred_points_action": pred_points_action}

    def compute_loss(self, model_output, batch, log_values={}, loss_prefix='', heads=None):
        points_action = batch['points_action'][:, :, :3]
        points_anchor = batch['points_anchor'][:, :, :3]
        points_trans_action = batch['points_action_trans'][:, :, :3]
        points_trans_anchor = batch['points_anchor_trans'][:, :, :3]
        
        N = points_action.shape[1]
        
        goal_emb = model_output["goal_emb"]

        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_output:
            ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
            points_action = torch.take_along_dim(points_action, ixs_action, dim=1)
            points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )

        if "sampled_ixs_anchor" in model_output:
            ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
            points_anchor = torch.take_along_dim(points_anchor, ixs_anchor, dim=1)
            points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )

        T0 = Transform3d(matrix=batch['T0'])
        T1 = Transform3d(matrix=batch['T1'])
        T_aug_list = [Transform3d(matrix=T_aug) for T_aug in batch['T_aug_list']] if 'T_aug_list' in batch else None

        if self.flow_supervision == "both":
            # Extract the flow and weights
            x_action = model_output["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            x_anchor = model_output["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            
            # Extract the predicted transform from the bidirectional flow and weights
            pred_T_action = dualflow2pose(
                xyz_src=points_trans_action, 
                xyz_tgt=points_trans_anchor,
                flow_src=pred_flow_action,
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action, 
                weights_tgt=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature
            )

            # Get induced flow and the transformed action points
            induced_flow_action = (
                pred_T_action.transform_points(points_trans_action) - 
                points_trans_action
            ).detach()
            pred_points_action = pred_T_action.transform_points(points_trans_action)

            # Get the ground truth transform
            # pred_T_action=T1T0^-1
            gt_T_action = T0.inverse().compose(T1)
            points_action_target = T1.transform_points(points_action)

            # Action losses
            # Loss associated with ground truth transform
            point_loss_action = mse_criterion(
                pred_points_action,
                points_action_target,
            )

            # Loss associated flow vectors matching a consistent rigid transform
            smoothness_loss_action = mse_criterion(
                pred_flow_action,
                induced_flow_action,
            )
            dense_loss_action = dense_flow_loss(
                points=points_trans_action,
                flow_pred=pred_flow_action,
                trans_gt=gt_T_action
            )
            
            # Anchor losses
            pred_T_anchor = pred_T_action.inverse()
            
            # Get the induced flow and the transformed anchor points
            induced_flow_anchor = (
                pred_T_anchor.transform_points(points_trans_anchor)
                - points_trans_anchor
            ).detach()
            pred_points_anchor = pred_T_anchor.transform_points(points_trans_anchor)
            
            # Get the ground truth transform
            gt_T_anchor = T1.inverse().compose(T0)
            points_anchor_target = T0.transform_points(points_anchor)
            
            # Loss associated with ground truth transform
            point_loss_anchor = mse_criterion(
                pred_points_anchor,
                points_anchor_target,
            )
            
            # Loss associated flow vectors matching a consistent rigid transform
            smoothness_loss_anchor = mse_criterion(
                pred_flow_anchor,
                induced_flow_anchor,
            )
            dense_loss_anchor = dense_flow_loss(
                points=points_trans_anchor,
                flow_pred=pred_flow_anchor,
                trans_gt=gt_T_anchor
            )
            
            self.action_weight = (self.action_weight) / (self.action_weight + self.anchor_weight)
            self.anchor_weight = (self.anchor_weight) / (self.action_weight + self.anchor_weight)
        elif self.flow_supervision == "action2anchor":
            # Extract the flow and weights
            x_action = model_output["flow_action"]
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            # Extract the predicted transform from the action->anchor flow and weights
            pred_T_action = flow2pose(
                xyz=points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature
            )
            
            # Action losses
            # Get induced flow and the transformed action points
            induced_flow_action = (
                pred_T_action.transform_points(points_trans_action)
                - points_trans_action
            ).detach()
            pred_points_action = pred_T_action.transform_points(points_trans_action)
            
            # Get the ground truth transform
            # pred_T_action=T1T0^-1
            gt_T_action = T0.inverse().compose(T1)
            points_action_target = T1.transform_points(points_action)
            
            # Loss associated with ground truth transform
            point_loss_action = mse_criterion(
                pred_points_action,
                points_action_target,
            )
            
            # Loss associated flow vectors matching a consistent rigid transform
            smoothness_loss_action = mse_criterion(
                pred_flow_action,
                induced_flow_action,
            )
            dense_loss_action = dense_flow_loss(
                points=points_trans_action,
                flow_pred=pred_flow_action,
                trans_gt=gt_T_action,
            )
        elif self.flow_supervision == "anchor2action":
            # Extract the flow and weights
            x_anchor = model_output["flow_anchor"]
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            
            # Extract the predicted transform from the anchor->action flow and weights
            pred_T_anchor = flow2pose(
                xyz=points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature
            )
            
            # Anchor losses
            # Get the induced flow and the transformed anchor points
            induced_flow_anchor = (
                pred_T_anchor.transform_points(points_trans_anchor)
                - points_trans_anchor
            ).detach()
            pred_points_anchor = pred_T_anchor.transform_points(points_trans_anchor)
            
            # Get the ground truth transform
            gt_T_anchor = T1.inverse().compose(T0)
            points_anchor_target = T0.transform_points(points_anchor)
            
            # Loss associated with ground truth transform
            point_loss_anchor = mse_criterion(
                pred_points_anchor,
                points_anchor_target,
            )
            
            # Loss associated flow vectors matching a consistent rigid transform
            smoothness_loss_anchor = mse_criterion(
                pred_flow_anchor,
                induced_flow_anchor,
            )
            dense_loss_anchor = dense_flow_loss(
                points=points_trans_anchor,
                flow_pred=pred_flow_anchor,
                trans_gt=gt_T_anchor,
            )
            
            # Action losses
            pred_T_action = pred_T_anchor.inverse()
            self.action_weight = 0
            self.anchor_weight = 1
            point_loss_action = 0
            smoothness_loss_action = 0
            dense_loss_action = 0
        else:
            raise ValueError(f"ERROR: Invalid flow supervision type: {self.flow_supervision}")    
        
        point_loss = (
            self.action_weight * point_loss_action
            + self.anchor_weight * point_loss_anchor
        )
        dense_loss = (
            self.action_weight * dense_loss_action
            + self.anchor_weight * dense_loss_anchor
        )
        smoothness_loss = (
            self.action_weight * smoothness_loss_action
            + self.anchor_weight * smoothness_loss_anchor
        )

        loss = (
            self.displace_weight * point_loss
            + self.smoothness_weight * smoothness_loss
            + self.consistency_weight * dense_loss
        )
            
        log_values[loss_prefix+'point_loss'] = self.displace_weight * point_loss
        log_values[loss_prefix+'smoothness_loss'] = self.smoothness_weight * smoothness_loss
        log_values[loss_prefix+'dense_loss'] = self.consistency_weight * dense_loss

        # Calculate error metrics compared to the demo
        if not self.min_err_across_racks_debug:
            pass
        else:
            action_center = None
            anchor_center = None
            
            error_R_mean, error_t_mean = get_2rack_errors(
                pred_T_action=pred_T_action, 
                T0=T0, 
                T1=T1, 
                mode=self.error_mode_2rack, 
                T_aug_list=T_aug_list, 
                action_center=action_center, 
                anchor_center=anchor_center
            )
            
            log_values[loss_prefix+'error_R_mean'] = error_R_mean
            log_values[loss_prefix+'error_t_mean'] = error_t_mean
            log_values[loss_prefix+'rotation_loss'] = self.rotation_weight * error_R_mean

        # Prior losses
        if self.model.conditioning in ["uniform_prior_pos_delta_l2norm"]:
            # Apply uniform prior to the goal embedding
            
            uniform = (torch.ones((goal_emb.shape[0], goal_emb.shape[1], N)) / goal_emb.shape[-1]).cuda().detach()
            action_kl = F.kl_div(F.log_softmax(uniform, dim=-1),
                                            F.log_softmax(goal_emb[:, :, :N], dim=-1), log_target=True,
                                            reduction='batchmean')
            anchor_kl = F.kl_div(F.log_softmax(uniform, dim=-1),
                                            F.log_softmax(goal_emb[:, :, N:], dim=-1), log_target=True,
                                            reduction='batchmean')
            vae_reg_loss = action_kl + anchor_kl
            loss += self.vae_reg_loss_weight * vae_reg_loss
            log_values[loss_prefix+'vae_reg_loss'] = self.vae_reg_loss_weight * vae_reg_loss

        # If using a continuous latent, apply a VAE regularization loss
        if heads is not None:
            def vae_regularization_loss(mu, log_var):
                # From https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/cvae.py#LL144C9-L144C105
                return torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim = 1).mean(dim = -1), dim = 0)
            
            # Regularize the global latent z to be standard normal
            if self.model.conditioning in ["latent_z_linear_internalcond"]:
                vae_reg_loss = vae_regularization_loss(heads['goal_emb_mu'], heads['goal_emb_logvar'])
                vae_reg_loss = torch.nan_to_num(vae_reg_loss)
                
                loss += self.vae_reg_loss_weight * vae_reg_loss
                log_values[loss_prefix+'vae_reg_loss'] = self.vae_reg_loss_weight * vae_reg_loss
            else:
                raise ValueError("ERROR: Why is there a non-None heads variable passed in when the model isn't even a latent_z model?")

        return loss, log_values

    def extract_flow_and_weight(self, x):
        # x: Batch, num_points, 4
        pred_flow = x[:, :, :3]
        if(x.shape[2] > 3):
            if self.sigmoid_on:
                pred_w = torch.sigmoid(x[:, :, 3])
            else:
                pred_w = x[:, :, 3]
        else:
            pred_w = None
        return pred_flow, pred_w

    def module_step(self, batch, batch_idx, log_prefix=''):
        points_trans_action = batch['points_action_trans']
        points_trans_anchor = batch['points_anchor_trans']
        points_action = batch['points_action']
        points_anchor = batch['points_anchor']

        # Model forward pass
        model_outputs = self.model(points_trans_action, 
                                   points_trans_anchor, 
                                   points_action,
                                   points_anchor, 
                                   n_samples=self.n_samples)

        # Extract from the model outputs
        # TODO only pass in points_anchor and points_action if the model is training
        if self.model.conditioning not in ["latent_z_linear_internalcond"]:
            heads = None
        else:
            heads = {'goal_emb_mu': model_outputs[0]['goal_emb_mu'], 'goal_emb_logvar': model_outputs[0]['goal_emb_logvar']}
        
        # Compute the p(z|Y) losses
        log_values = {}
        loss, log_values = self.compute_loss(
            model_outputs[0], batch, log_values=log_values, loss_prefix=log_prefix, heads=heads)
        
        # Debugging, plot inference errors when sampling from known prior, and get errors when sampling more than once
        if self.get_sample_errors:
            with torch.no_grad():
                def get_inference_error(log_values, batch, loss_prefix):
                    T0 = Transform3d(matrix=batch['T0'])
                    T1 = Transform3d(matrix=batch['T1'])
                    T_aug_list = [Transform3d(matrix=T_aug) for T_aug in batch['T_aug_list']] if 'T_aug_list' in batch else None

                    if self.model.conditioning not in ["uniform_prior_pos_delta_l2norm", "latent_z_linear_internalcond"]:
                        model_outputs = self.model(points_trans_action, points_trans_anchor, points_action, points_anchor, mode="forward", n_samples=1)
                    else:
                        model_outputs = self.model(points_trans_action, points_trans_anchor, points_action, points_anchor, mode="inference", n_samples=1)
                        
                    x_action = model_outputs[0]['flow_action']
                    x_anchor = model_outputs[0]['flow_anchor']
                    goal_emb = model_outputs[0]['goal_emb']

                    
                    # If we've applied some sampling, we need to extract the predictions too...
                    if "sampled_ixs_action" in model_outputs[0]:
                        ixs_action = model_outputs[0]["sampled_ixs_action"].unsqueeze(-1)
                        sampled_points_trans_action = torch.take_along_dim(
                            points_trans_action, ixs_action, dim=1
                        )
                    else:
                        sampled_points_trans_action = points_trans_action

                    if "sampled_ixs_anchor" in model_outputs[0]:
                        ixs_anchor = model_outputs[0]["sampled_ixs_anchor"].unsqueeze(-1)
                        sampled_points_trans_anchor = torch.take_along_dim(
                            points_trans_anchor, ixs_anchor, dim=1
                        )
                    else:
                        sampled_points_trans_anchor = points_trans_anchor
                    
                    pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
                    pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
                    
                    del x_action, x_anchor, goal_emb

                    pred_T_action = dualflow2pose(xyz_src=sampled_points_trans_action, 
                                                xyz_tgt=sampled_points_trans_anchor,
                                                flow_src=pred_flow_action, 
                                                flow_tgt=pred_flow_anchor,
                                                weights_src=pred_w_action, 
                                                weights_tgt=pred_w_anchor,
                                                return_transform3d=True, 
                                                normalization_scehme=self.weight_normalize,
                                                temperature=self.softmax_temperature)

                    if not self.min_err_across_racks_debug:
                        # don't print rotation/translation error metrics to the logs
                        pass
                    else:
                        error_R_mean, error_t_mean = get_2rack_errors(pred_T_action, T0, T1, mode=self.error_mode_2rack, T_aug_list=T_aug_list)
                        log_values[loss_prefix+'sample_error_R_mean'] = error_R_mean
                        log_values[loss_prefix+'sample_error_t_mean'] = error_t_mean
                get_inference_error(log_values, batch, loss_prefix=log_prefix)
                        
        return loss, log_values

    def visualize_results(self, batch, batch_idx, log_prefix=''):
        points_trans_action = batch['points_action_trans']
        points_trans_anchor = batch['points_anchor_trans']
        points_action = batch['points_action']
        points_anchor = batch['points_anchor']

        T1 = Transform3d(matrix=batch['T1'])

        # Model forward pass
        model_outputs = self.model(
            points_trans_action, 
            points_trans_anchor, 
            points_action, 
            points_anchor, 
            n_samples=1
        )
        # Extract from the model outputs
        goal_emb = model_outputs[0]['goal_emb']

        # Only use XYZ
        points_action = points_action[:, :, :3]
        points_anchor = points_anchor[:, :, :3]
        points_trans_action = points_trans_action[:, :, :3]
        points_trans_anchor = points_trans_anchor[:, :, :3]

        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_outputs[0]:
            ixs_action = model_outputs[0]["sampled_ixs_action"].unsqueeze(-1)
            sampled_points_action = torch.take_along_dim(
                points_action, ixs_action, dim=1
            )
            sampled_points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )
        else:
            sampled_points_action = points_action
            sampled_points_trans_action = points_trans_action

        if "sampled_ixs_anchor" in model_outputs[0]:
            ixs_anchor = model_outputs[0]["sampled_ixs_anchor"].unsqueeze(-1)
            sampled_points_anchor = torch.take_along_dim(
                points_anchor, ixs_anchor, dim=1
            )
            sampled_points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )
        else:
            sampled_points_anchor = points_anchor
            sampled_points_trans_anchor = points_trans_anchor

        # Get predicted transform
        if self.flow_supervision == "both":
            # Extract flow and weights
            x_action = model_outputs[0]['flow_action']
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            x_anchor = model_outputs[0]['flow_anchor']
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            
            pred_T_action = dualflow2pose(
                xyz_src=sampled_points_trans_action, 
                xyz_tgt=sampled_points_trans_anchor,
                flow_src=pred_flow_action, 
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action, 
                weights_tgt=pred_w_anchor,
                return_transform3d=True, 
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature
            )
        elif self.flow_supervision == "action2anchor":
            # Extract flow and weights
            x_action = model_outputs[0]['flow_action']
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            pred_T_action = flow2pose(
                xyz=sampled_points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature
            )
        elif self.flow_supervision == "anchor2action":
            # Extract flow and weights
            x_anchor = model_outputs[0]['flow_anchor']
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            
            pred_T_anchor = flow2pose(
                xyz=sampled_points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.weight_normalize,
                temperature=self.softmax_temperature
            )
            pred_T_action = pred_T_anchor.inverse()
        else:
            raise ValueError(f"ERROR: Invalid flow supervision type: {self.flow_supervision}")

        pred_points_action = pred_T_action.transform_points(
            points_trans_action)

        # Logging results
        res_images = {}

        demo_points_tensors = [points_action[0], points_anchor[0]]
        demo_points_colors = ['blue', 'red']
        if 'points_action_aug_trans' in batch:
            demo_points_tensors.append(batch['points_action_aug_trans'][0, :, :3])
            demo_points_colors.append('yellow')
        demo_points = get_color(
            tensor_list=demo_points_tensors, color_list=demo_points_colors)
        res_images[log_prefix+'demo_points'] = wandb.Object3D(
            demo_points)

        action_transformed_action = get_color(
            tensor_list=[points_action[0], points_trans_action[0]], color_list=['blue', 'red'])
        res_images[log_prefix+'action_transformed_action'] = wandb.Object3D(
            action_transformed_action)

        anchor_transformed_anchor = get_color(
            tensor_list=[points_anchor[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        res_images[log_prefix+'anchor_transformed_anchor'] = wandb.Object3D(
            anchor_transformed_anchor)

        demo_points_apply_action_transform = get_color(
            tensor_list=[pred_points_action[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        res_images[log_prefix+'demo_points_apply_action_transform'] = wandb.Object3D(
            demo_points_apply_action_transform)

        apply_action_transform_demo_comparable = get_color(
            tensor_list=[T1.inverse().transform_points(pred_points_action)[0], T1.inverse().transform_points(points_trans_anchor)[0]], color_list=['blue', 'red'])
        res_images[log_prefix+'apply_action_transform_demo_comparable'] = wandb.Object3D(
            apply_action_transform_demo_comparable)

        predicted_vs_gt_transform_tensors = [T1.inverse().transform_points(pred_points_action)[0], points_action[0], T1.inverse().transform_points(points_trans_anchor)[0]]
        predicted_vs_gt_transform_colors = ['blue', 'green', 'red']
        if 'points_action_aug_trans' in batch:
            predicted_vs_gt_transform_tensors.append(batch['points_action_aug_trans'][0, :, :3])
            predicted_vs_gt_transform_colors.append('yellow')
        predicted_vs_gt_transform_applied = get_color(
            tensor_list=predicted_vs_gt_transform_tensors, color_list=predicted_vs_gt_transform_colors)
        res_images[log_prefix+'predicted_vs_gt_transform_applied'] = wandb.Object3D(
            predicted_vs_gt_transform_applied)

        apply_predicted_transform = get_color(
            tensor_list=[T1.inverse().transform_points(pred_points_action)[0], T1.inverse().transform_points(points_trans_action)[0], T1.inverse().transform_points(points_trans_anchor)[0]], color_list=['blue', 'orange', 'red', ])
        res_images[log_prefix+'apply_predicted_transform'] = wandb.Object3D(
            apply_predicted_transform)

        pred_w_points_list = []
        pred_w_colors_list = []
        if self.flow_supervision in ["both", "action2anchor"]:
            pred_w_points_list.append(sampled_points_action[0].detach())
            pred_w_colors_list.append(color_gradient(pred_w_action[0]))
        if self.flow_supervision in ["both", "anchor2action"]:
            pred_w_points_list.append(sampled_points_anchor[0].detach())
            pred_w_colors_list.append(color_gradient(pred_w_anchor[0]))    
        
        pred_w_on_objects = np.concatenate([
            torch.cat(pred_w_points_list, dim=0).cpu().numpy(),
            np.concatenate(pred_w_colors_list, axis=0)],
            axis=-1)
        
        res_images[log_prefix+'pred_w'] = wandb.Object3D(
            pred_w_on_objects, markerSize=1000)

        # This visualization only applies to methods that have discrete per-point latents
        if self.model.conditioning not in ["latent_z_linear_internalcond"]:
            # Plot goal embeddings on objects
            goal_emb_norm_action = F.softmax(goal_emb[0, :, :points_action.shape[1]], dim=-1).detach().cpu()
            goal_emb_norm_anchor = F.softmax(goal_emb[0, :, points_action.shape[1]:], dim=-1).detach().cpu()
            colors_action = color_gradient(goal_emb_norm_action[0])
            colors_anchor = color_gradient(goal_emb_norm_anchor[0])
            goal_emb_on_objects = np.concatenate([
                torch.cat([points_action[0].detach(), points_anchor[0].detach()], dim=0).cpu().numpy(),
                np.concatenate([colors_action, colors_anchor], axis=0)],
                axis=-1)
            res_images[log_prefix+'goal_emb'] = wandb.Object3D(
                goal_emb_on_objects)

            # Plot the p(z|Y) embeddings on the objects
            if self.plot_encoder_distribution:
                # Get embeddings for p(z|Y2)
                if 'points_action_aug_trans' in batch:
                    points_action_aug_trans = batch['points_action_aug_trans']
                    points_trans_action = batch['points_action_trans']
                    points_trans_anchor = batch['points_anchor_trans']
                    points_action = batch['points_action']
                    points_anchor = batch['points_anchor']
                    
                    assert points_action_aug_trans.shape[1] == points_anchor.shape[1], \
                        f"ERROR: Augmented action points have different number of points than the anchor points. Are you using more than 1 distractor?"
                    
                    model_no_cond_x_outputs = self.model(points_trans_action, 
                                                         points_trans_anchor, 
                                                         points_action_aug_trans, 
                                                         points_anchor, 
                                                         n_samples=1)
                        
                    pzY2_emb = model_no_cond_x_outputs[0]['goal_emb']
                    
                    pzY2_action_emb = pzY2_emb[0, :, :points_action.shape[1]]
                    pzY2_anchor_emb = pzY2_emb[0, :, points_action.shape[1]:]
                    pzY2_action_dist = F.softmax(pzY2_action_emb, dim=-1).detach().cpu().numpy()[0]
                    pzY2_anchor_dist = F.softmax(pzY2_anchor_emb, dim=-1).detach().cpu().numpy()[0]
                    
                    pzY2_actions_probs = np.array([prob for prob in pzY2_action_dist])
                    pzY2_anchor_probs = np.array([prob for prob in pzY2_anchor_dist])
                
                pzY1_action_emb = goal_emb[0, :, :points_action.shape[1]]
                pzY1_anchor_emb = goal_emb[0, :, points_action.shape[1]:]
                pzY1_action_dist = F.softmax(pzY1_action_emb, dim=-1).detach().cpu().numpy()[0]
                pzY1_anchor_dist = F.softmax(pzY1_anchor_emb, dim=-1).detach().cpu().numpy()[0]
                
                x_vals = np.arange(pzY1_action_dist.shape[0])
                pzY1_actions_probs = np.array([prob for prob in pzY1_action_dist])
                pzY1_anchor_probs = np.array([prob for prob in pzY1_anchor_dist])

                layout_margin = go.layout.Margin(
                    l=50, #left margin
                    r=120, #right margin
                    b=50, #bottom margin
                    t=50, #top margin
                    autoexpand=False 
                )

                # Plot action distributions
                action_max_prob = np.max(pzY1_actions_probs)
                
                pzY_action_data = [
                    go.Bar(name='pzY1', x=x_vals, y=pzY1_actions_probs, width=1, marker_color='blue', opacity=0.5, showlegend=True)
                ]
                if 'points_action_aug_trans' in batch:
                    pzY_action_data.append(
                        go.Bar(name='pzY2', x=x_vals, y=pzY2_actions_probs, width=1, marker_color='red', opacity=0.5, showlegend=True)
                    )
                    action_max_prob = max(action_max_prob, np.max(pzY2_actions_probs))
                
                pzY_action_plot = go.Figure(data=pzY_action_data)                
                pzY_action_plot.update_layout(barmode='overlay', 
                                              height=480, 
                                              width=1920, 
                                              yaxis_range=[0, action_max_prob*1.1], 
                                              margin=layout_margin,
                                              legend={'entrywidth': 40})
                
                # Plot anchor distributions
                anchor_max_prob = np.max(pzY1_anchor_probs)
                
                pzY_anchor_data = [
                    go.Bar(name='pzY1', x=x_vals, y=pzY1_anchor_probs, width=1, marker_color='blue', opacity=0.5, showlegend=True)
                ]
                if 'points_action_aug_trans' in batch:
                    pzY_anchor_data.append(
                        go.Bar(name='pzY2', x=x_vals, y=pzY2_anchor_probs, width=1, marker_color='red', opacity=0.5, showlegend=True)
                    )
                    anchor_max_prob = max(anchor_max_prob, np.max(pzY2_anchor_probs))
                    
                pzY_anchor_plot = go.Figure(data=pzY_anchor_data)
                pzY_anchor_plot.update_layout(barmode='overlay', 
                                              height=480, 
                                              width=1920, 
                                              yaxis_range=[0, anchor_max_prob*1.1], 
                                              margin=layout_margin,
                                              legend={'entrywidth': 40})
                
                res_images[log_prefix+'pzY_action_distribution'] = wandb.Html(plotly.io.to_html(pzY_action_plot, include_plotlyjs='cdn'))
                res_images[log_prefix+'pzY_anchor_distribution'] = wandb.Html(plotly.io.to_html(pzY_anchor_plot, include_plotlyjs='cdn'))

        return res_images



class EquivarianceTrainingModule_WithPZCondX(PointCloudTrainingModule):

    def __init__(self,
                 model_with_cond_x,
                 training_module_no_cond_x,
                 goal_emb_cond_x_loss_weight=1,
                 freeze_residual_flow=False,
                 freeze_z_embnn=False,
                 freeze_embnn=False,
                 plot_encoder_distribution=False,
                 goal_emb_cond_x_loss_type="forward_kl",
                 overwrite_loss=False,
    ):

        super().__init__(model=model_with_cond_x, lr=training_module_no_cond_x.lr, 
                         image_log_period=training_module_no_cond_x.image_log_period)

        self.model_with_cond_x = model_with_cond_x
        self.model = self.model_with_cond_x.residflow_embnn
        self.training_module_no_cond_x = training_module_no_cond_x
        self.goal_emb_cond_x_loss_weight = goal_emb_cond_x_loss_weight
        self.goal_emb_cond_x_loss_type = goal_emb_cond_x_loss_type
        
        self.cfg_freeze_residual_flow = freeze_residual_flow
        self.cfg_freeze_z_embnn = freeze_z_embnn
        self.cfg_freeze_embnn = freeze_embnn
        
        self.n_samples = 1
        self.plot_encoder_distribution = plot_encoder_distribution
        
        self.overwrite_loss = overwrite_loss or (self.cfg_freeze_embnn and self.cfg_freeze_z_embnn and self.cfg_freeze_residual_flow)

        self.flow_supervision = self.training_module_no_cond_x.flow_supervision

    def action_centered(self, points_action, points_anchor):
        """
        @param points_action, (1,num_points,3)
        @param points_anchor, (1,num_points,3)
        """
        points_action_mean = points_action.clone().mean(axis=1)
        points_action_mean_centered = points_action-points_action_mean
        points_anchor_mean_centered = points_anchor - points_action_mean

        return points_action_mean_centered, points_anchor_mean_centered, points_action_mean

    def get_transform(self, points_trans_action, points_trans_anchor, points_action=None, points_anchor=None, mode="forward", sampling_method="gumbel", n_samples=1):
        # mode is unused
        model_outputs = self.model_with_cond_x(points_trans_action, 
                                               points_trans_anchor, 
                                               points_action, 
                                               points_anchor, 
                                               sampling_method=sampling_method, 
                                               n_samples=n_samples)

        ans_dicts = []
        for i in range(n_samples):
            model_output = model_outputs[i]
            goal_emb = model_output['goal_emb']
            goal_emb_cond_x = model_output['goal_emb_cond_x']

            points_trans_action = points_trans_action[:, :, :3]
            points_trans_anchor = points_trans_anchor[:, :, :3]
            ans_dict = self.predict(model_output=model_output,
                                    points_trans_action=points_trans_action, points_trans_anchor=points_trans_anchor)
            
            ans_dict['flow_components'] = model_output
            if self.model_with_cond_x.return_debug:
                for_debug['goal_emb'] = goal_emb
                for_debug['goal_emb_cond_x'] = goal_emb_cond_x
                ans_dict['for_debug'] = for_debug
            ans_dicts.append(ans_dict)
        return ans_dicts

    def predict(self, model_output, points_trans_action, points_trans_anchor):
        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_output:
            ixs_action = model_output["sampled_ixs_action"].unsqueeze(-1)
            sampled_points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )
        else:
            sampled_points_trans_action = points_trans_action

        if "sampled_ixs_anchor" in model_output:
            ixs_anchor = model_output["sampled_ixs_anchor"].unsqueeze(-1)
            sampled_points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )
        else:
            sampled_points_trans_anchor = points_trans_anchor

        if self.flow_supervision == "both":
            # Extract flow and weights
            x_action = model_output['flow_action']
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            x_anchor = model_output['flow_anchor']
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
        
            pred_T_action = dualflow2pose(
                xyz_src=sampled_points_trans_action, 
                xyz_tgt=sampled_points_trans_anchor,
                flow_src=pred_flow_action, 
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action, 
                weights_tgt=pred_w_anchor,
                return_transform3d=True, 
                normalization_scehme=self.training_module_no_cond_x.weight_normalize, 
                temperature=self.training_module_no_cond_x.softmax_temperature
            )
        elif self.flow_supervision == "action2anchor":
            # Extract flow and weights
            x_action = model_output['flow_action']
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            # Get predicted transform
            pred_T_action = flow2pose(
                xyz=sampled_points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature
            )
        elif self.flow_supervision == "anchor2action":
            # Extract flow and weights
            x_anchor = model_output['flow_anchor']
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            
            pred_T_anchor = flow2pose(
                xyz=sampled_points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature
            )
            pred_T_action = pred_T_anchor.inverse()
        else:
            raise ValueError(f"ERROR: Invalid flow supervision type: {self.flow_supervision}")
 
        pred_points_action = pred_T_action.transform_points(points_trans_action)

        return {"pred_T_action": pred_T_action,
                "pred_points_action": pred_points_action}

    def compute_loss(self, model_output, batch, log_values={}, loss_prefix=''):
        N_action = batch['points_action'].shape[1]
        N_anchor = batch['points_anchor'].shape[1]
        
        goal_emb = model_output['goal_emb']
        goal_emb_cond_x = model_output['goal_emb_cond_x']
        
        # Calculate p(z|Y) losses using p(z|X) model outputs, usually just for logging
        loss, log_values = self.training_module_no_cond_x.compute_loss(model_output, batch, log_values, loss_prefix)
        # aka "if it is training time and not val time"
        if goal_emb is not None:
            # Calculate losses between p(z|Y) and p(z|X)
            if self.model_with_cond_x.conditioning in ['pos_delta_l2norm', 'uniform_prior_pos_delta_l2norm']:
                B, K, D = goal_emb.shape
                if self.goal_emb_cond_x_loss_type == "forward_kl":
                    action_kl = F.kl_div(F.log_softmax(goal_emb_cond_x[:, :, :N_action], dim=-1),
                                         F.log_softmax(goal_emb[:, :, :N_action], dim=-1), 
                                         log_target=True,
                                         reduction='batchmean')
                    anchor_kl = F.kl_div(F.log_softmax(goal_emb_cond_x[:, :, N_action:], dim=-1),
                                         F.log_softmax(goal_emb[:, :, N_action:], dim=-1), 
                                         log_target=True,
                                         reduction='batchmean')
                    
                elif self.goal_emb_cond_x_loss_type == "reverse_kl":
                    action_kl = F.kl_div(F.log_softmax(goal_emb[:, :, :N_action], dim=-1),
                                         F.log_softmax(goal_emb_cond_x[:, :, :N_action], dim=-1), 
                                         log_target=True,
                                         reduction='batchmean')
                    anchor_kl = F.kl_div(F.log_softmax(goal_emb[:, :, N_action:], dim=-1),
                                         F.log_softmax(goal_emb_cond_x[:, :, N_action:], dim=-1), 
                                         log_target=True,
                                         reduction='batchmean')
                elif self.goal_emb_cond_x_loss_type in ["js_div"]:
                    eps = 1e-8
                    
                    action_kl = js_div(q=F.log_softmax(goal_emb_cond_x[:, :, :N_action], dim=-1),
                                       p=F.log_softmax(goal_emb[:, :, :N_action], dim=-1),
                                       reduction='batchmean',
                                       eps=eps)
                    anchor_kl = js_div(q=F.log_softmax(goal_emb_cond_x[:, :, N_action:], dim=-1),
                                       p=F.log_softmax(goal_emb[:, :, N_action:], dim=-1),
                                       reduction='batchmean',
                                       eps=eps)
                elif self.goal_emb_cond_x_loss_type == "mse":
                    action_kl = F.mse_loss(F.softmax(goal_emb_cond_x[:, :, :N_action], dim=-1),
                                           F.softmax(goal_emb[:, :, :N_action], dim=-1), 
                                           reduction='sum')
                                           
                    anchor_kl = F.mse_loss(F.softmax(goal_emb_cond_x[:, :, N_action:], dim=-1),
                                           F.softmax(goal_emb[:, :, N_action:], dim=-1), 
                                           reduction='sum')
                else:
                    raise ValueError(f'goal_emb_cond_x_loss_type={self.goal_emb_cond_x_loss_type} not supported')

            elif self.model_with_cond_x.conditioning in ["latent_z_linear_internalcond"]:
                if self.goal_emb_cond_x_loss_type == "forward":
                    # TODO fix this to not just be the mean
                    input_emb = goal_emb_cond_x[0] # just take the mean
                    target_emb = goal_emb

                    action_kl = F.kl_div(F.log_softmax(input_emb[:, :, 0], dim=-1),
                                            F.log_softmax(target_emb[:, :, 0], dim=-1), log_target=True,
                                            reduction='batchmean')
                    anchor_kl = F.kl_div(F.log_softmax(input_emb[:, :, 1], dim=-1),
                                            F.log_softmax(target_emb[:, :, 1], dim=-1), log_target=True,
                                            reduction='batchmean')
                elif self.goal_emb_cond_x_loss_type == "reverse":
                    # TODO fix this to not just be the mean
                    input_emb = goal_emb
                    target_emb = goal_emb_cond_x[0] # just take the mean

                    action_kl = F.kl_div(F.log_softmax(input_emb[:, :, 0], dim=-1),
                                            F.log_softmax(target_emb[:, :, 0], dim=-1), log_target=True,
                                            reduction='batchmean')
                    anchor_kl = F.kl_div(F.log_softmax(input_emb[:, :, 1], dim=-1),
                                            F.log_softmax(target_emb[:, :, 1], dim=-1), log_target=True,
                                            reduction='batchmean')
                elif self.goal_emb_cond_x_loss_type in ["js_div"]:
                    def compute_js_loss(source_mu, source_log_var, target_mu, target_log_var):
                        # From https://discuss.pytorch.org/t/compute-js-loss-between-gaussian-distributions-parameterized-by-mu-and-log-var/130935
                        def get_prob(mu, log_var):
                            dist = Normal(mu, torch.exp(0.5 * log_var))
                            val = dist.sample()
                            return dist.log_prob(val).exp()

                        def kl_loss(p, q):
                            return F.kl_div(p, q, reduction="batchmean", log_target=False)

                        source_prob = get_prob(source_mu, source_log_var)
                        target_prob = get_prob(target_mu, target_log_var)

                        log_mean_prob = (0.5 * (source_prob + target_prob)).log()
                        js_loss = 0.5 * (kl_loss(log_mean_prob, source_prob) + kl_loss(log_mean_prob, target_prob))
                        return js_loss

                    ACTION = 0
                    ANCHOR = 1
                    action_kl = compute_js_loss(goal_emb[0][:,:,ACTION], goal_emb[1][:,:,ACTION], goal_emb_cond_x[0][:,:,ACTION], goal_emb_cond_x[1][:,:,ACTION])
                    anchor_kl = compute_js_loss(goal_emb[0][:,:,ANCHOR], goal_emb[1][:,:,ANCHOR], goal_emb_cond_x[0][:,:,ANCHOR], goal_emb_cond_x[1][:,:,ANCHOR])
                else:
                    raise ValueError("ERROR: goal_emb_cond_x_loss_type must be forward or reverse or js")
            else:
                raise ValueError(f'conditioning={self.model_with_cond_x.conditioning} not supported')
            
            
            goal_emb_loss = action_kl + anchor_kl

            if self.overwrite_loss:
                # Only update p(z|X) encoder for p(z|X) pass
                loss = self.goal_emb_cond_x_loss_weight * goal_emb_loss
            else:
                # Update p(z|X) encoder and TAXPose decoder for p(z|X) pass
                loss += self.goal_emb_cond_x_loss_weight * goal_emb_loss

            log_values[loss_prefix+'goal_emb_cond_x_loss'] = self.goal_emb_cond_x_loss_weight * goal_emb_loss
            log_values[loss_prefix+'action_kl'] = action_kl
            log_values[loss_prefix+'anchor_kl'] = anchor_kl

        return loss, log_values

    def extract_flow_and_weight(self, *args, **kwargs):
        return self.training_module_no_cond_x.extract_flow_and_weight(*args, **kwargs)

    def module_step(self, batch, batch_idx, log_prefix=''):
        points_trans_action = batch['points_action_trans']
        points_trans_anchor = batch['points_anchor_trans']
        points_action = batch['points_action']
        points_anchor = batch['points_anchor']

        # If joint training prior
        # Unfreeze components for p(z|Y) pass
        self.training_module_no_cond_x.model.freeze_residual_flow = False
        self.training_module_no_cond_x.model.freeze_z_embnn = False
        self.training_module_no_cond_x.model.freeze_embnn = False
        self.training_module_no_cond_x.model.tax_pose.freeze_embnn = False
        
        # p(z|Y) pass
        pzY_loss, pzY_log_values = self.training_module_no_cond_x.module_step(batch, batch_idx)
        
        # Potentially freeze components for p(z|X) pass
        self.training_module_no_cond_x.model.freeze_residual_flow = self.cfg_freeze_residual_flow
        self.training_module_no_cond_x.model.freeze_z_embnn = self.cfg_freeze_z_embnn
        self.training_module_no_cond_x.model.freeze_embnn = self.cfg_freeze_embnn
        self.training_module_no_cond_x.model.tax_pose.freeze_embnn = self.cfg_freeze_embnn
        
        # Debugging, optionally use the exact same z samples for p(z|X) as selected by p(z|Y)
        z_samples = None

        # Do the p(z|X) pass, determine whether continuous latent z is sampled
        model_outputs = self.model_with_cond_x(points_trans_action, 
                                               points_trans_anchor, 
                                               points_action, 
                                               points_anchor, 
                                               sampling_method="gumbel", 
                                               n_samples=self.n_samples,
                                               z_samples=z_samples)
        goal_emb = model_outputs[0]['goal_emb']
        goal_emb_cond_x = model_outputs[0]['goal_emb_cond_x']

        # Compute p(z|X) losses
        log_values = {}
        log_prefix = 'pzX_'
        loss, log_values = self.compute_loss(
            model_outputs[0], batch, log_values=log_values, loss_prefix=log_prefix)        
        
        # If joint training prior also use p(z|Y) losses
        loss = pzY_loss + loss
        log_values = {**pzY_log_values, **log_values}
        
        return loss, log_values
    
    def visualize_results(self, batch, batch_idx, log_prefix=''):
        res_images = self.training_module_no_cond_x.visualize_results(batch, batch_idx, log_prefix='pzY_')

        points_trans_action = batch['points_action_trans']
        points_trans_anchor = batch['points_anchor_trans']
        points_action = batch['points_action']
        points_anchor = batch['points_anchor']

        T1 = Transform3d(matrix=batch['T1'])

        # Debugging, optionally use the exact same z samples for p(z|X) as selected by p(z|Y)
        z_samples = None
        
        # Do the p(z|X) pass, determine whether continuous latent z is sampled
        model_outputs = self.model_with_cond_x(points_trans_action, 
                                                points_trans_anchor, 
                                                points_action, 
                                                points_anchor,
                                                sampling_method="gumbel",
                                                n_samples=1,
                                                z_samples=z_samples)
        goal_emb = model_outputs[0]['goal_emb']
        goal_emb_cond_x = model_outputs[0]['goal_emb_cond_x']

        points_action = points_action[:, :, :3]
        points_anchor = points_anchor[:, :, :3]
        points_trans_action = points_trans_action[:, :, :3]
        points_trans_anchor = points_trans_anchor[:, :, :3]

        # If we've applied some sampling, we need to extract the predictions too...
        if "sampled_ixs_action" in model_outputs[0]:
            ixs_action = model_outputs[0]["sampled_ixs_action"].unsqueeze(-1)
            sampled_points_action = torch.take_along_dim(
                points_action, ixs_action, dim=1
            )
            sampled_points_trans_action = torch.take_along_dim(
                points_trans_action, ixs_action, dim=1
            )
        else:
            sampled_points_action = points_action
            sampled_points_trans_action = points_trans_action

        if "sampled_ixs_anchor" in model_outputs[0]:
            ixs_anchor = model_outputs[0]["sampled_ixs_anchor"].unsqueeze(-1)
            sampled_points_anchor = torch.take_along_dim(
                points_anchor, ixs_anchor, dim=1
            )
            sampled_points_trans_anchor = torch.take_along_dim(
                points_trans_anchor, ixs_anchor, dim=1
            )
        else:
            sampled_points_anchor = points_anchor
            sampled_points_trans_anchor = points_trans_anchor

        if self.flow_supervision == "both":
            # Extract flow and weights
            x_action = model_outputs[0]['flow_action']
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            x_anchor = model_outputs[0]['flow_anchor']
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            
            pred_T_action = dualflow2pose(
                xyz_src=sampled_points_trans_action, 
                xyz_tgt=sampled_points_trans_anchor,
                flow_src=pred_flow_action, 
                flow_tgt=pred_flow_anchor,
                weights_src=pred_w_action, 
                weights_tgt=pred_w_anchor,
                return_transform3d=True, 
                normalization_scehme=self.training_module_no_cond_x.weight_normalize, 
                temperature=self.training_module_no_cond_x.softmax_temperature
            )
        elif self.flow_supervision == "action2anchor":
            # Extract flow and weights
            x_action = model_outputs[0]['flow_action']
            pred_flow_action, pred_w_action = self.extract_flow_and_weight(x_action)
            
            # Get predicted transform
            pred_T_action = flow2pose(
                xyz=sampled_points_trans_action,
                flow=pred_flow_action,
                weights=pred_w_action,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature
            )
        elif self.flow_supervision == "anchor2action":
            # Extract flow and weights
            x_anchor = model_outputs[0]['flow_anchor']
            pred_flow_anchor, pred_w_anchor = self.extract_flow_and_weight(x_anchor)
            
            pred_T_anchor = flow2pose(
                xyz=sampled_points_trans_anchor,
                flow=pred_flow_anchor,
                weights=pred_w_anchor,
                return_transform3d=True,
                normalization_scehme=self.training_module_no_cond_x.weight_normalize,
                temperature=self.training_module_no_cond_x.softmax_temperature
            )
            pred_T_action = pred_T_anchor.inverse()
        else:
            raise ValueError(f"ERROR: Invalid flow supervision type: {self.flow_supervision}")
        
        pred_points_action = pred_T_action.transform_points(points_trans_action)
        
        demo_points_tensors = [points_action[0], points_anchor[0]]
        demo_points_colors = ['blue', 'red']
        if 'points_action_aug_trans' in batch:
            demo_points_tensors.append(batch['points_action_aug_trans'][0, :, :3])
            demo_points_colors.append('yellow')
        demo_points = get_color(
            tensor_list=demo_points_tensors, color_list=demo_points_colors)
        res_images[log_prefix+'demo_points'] = wandb.Object3D(
            demo_points)

        action_transformed_action = get_color(
            tensor_list=[points_action[0], points_trans_action[0]], color_list=['blue', 'red'])
        res_images[log_prefix+'action_transformed_action'] = wandb.Object3D(
            action_transformed_action)

        anchor_transformed_anchor = get_color(
            tensor_list=[points_anchor[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        res_images[log_prefix+'anchor_transformed_anchor'] = wandb.Object3D(
            anchor_transformed_anchor)

        # transformed_input_points = get_color(tensor_list=[
        #     points_trans_action[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        # res_images[log_prefix+'transformed_input_points'] = wandb.Object3D(
        #     transformed_input_points)

        demo_points_apply_action_transform = get_color(
            tensor_list=[pred_points_action[0], points_trans_anchor[0]], color_list=['blue', 'red'])
        res_images[log_prefix+'demo_points_apply_action_transform'] = wandb.Object3D(
            demo_points_apply_action_transform)

        apply_action_transform_demo_comparable = get_color(
            tensor_list=[T1.inverse().transform_points(pred_points_action)[0], T1.inverse().transform_points(points_trans_anchor)[0]], color_list=['blue', 'red'])
        res_images[log_prefix+'apply_action_transform_demo_comparable'] = wandb.Object3D(
            apply_action_transform_demo_comparable)

        predicted_vs_gt_transform_tensors = [T1.inverse().transform_points(pred_points_action)[0], points_action[0], T1.inverse().transform_points(points_trans_anchor)[0]]
        predicted_vs_gt_transform_colors = ['blue', 'green', 'red']
        if 'points_action_aug_trans' in batch:
            predicted_vs_gt_transform_tensors.append(batch['points_action_aug_trans'][0, :, :3])
            predicted_vs_gt_transform_colors.append('yellow')
        predicted_vs_gt_transform_applied = get_color(
            tensor_list=predicted_vs_gt_transform_tensors, color_list=predicted_vs_gt_transform_colors)
        res_images[log_prefix+'predicted_vs_gt_transform_applied'] = wandb.Object3D(
            predicted_vs_gt_transform_applied)

        apply_predicted_transform = get_color(
            tensor_list=[T1.inverse().transform_points(pred_points_action)[0], T1.inverse().transform_points(points_trans_action)[0], T1.inverse().transform_points(points_trans_anchor)[0]], color_list=['blue', 'orange', 'red', ])
        res_images[log_prefix+'apply_predicted_transform'] = wandb.Object3D(
            apply_predicted_transform)

        pred_w_points_list = []
        pred_w_colors_list = []
        if self.flow_supervision in ["both", "action2anchor"]:
            pred_w_points_list.append(sampled_points_action[0].detach())
            pred_w_colors_list.append(color_gradient(pred_w_action[0]))
        if self.flow_supervision in ["both", "anchor2action"]:
            pred_w_points_list.append(sampled_points_anchor[0].detach())
            pred_w_colors_list.append(color_gradient(pred_w_anchor[0]))    
        
        pred_w_on_objects = np.concatenate([
            torch.cat(pred_w_points_list, dim=0).cpu().numpy(),
            np.concatenate(pred_w_colors_list, axis=0)],
            axis=-1)
        
        res_images[log_prefix+'pred_w'] = wandb.Object3D(
            pred_w_on_objects, markerSize=1000)

        # This visualization only applies to methods that have discrete per-point latents
        if self.model.conditioning not in ["latent_z_linear", "latent_z_linear_internalcond"]:
            # Plot goal embeddings on objects
            goal_emb_cond_x_norm_action = F.softmax(goal_emb_cond_x[0, :, :points_action.shape[1]], dim=-1).detach().cpu()
            goal_emb_cond_x_norm_anchor = F.softmax(goal_emb_cond_x[0, :, points_action.shape[1]:], dim=-1).detach().cpu()

            colors_action = color_gradient(goal_emb_cond_x_norm_action[0])
            colors_anchor = color_gradient(goal_emb_cond_x_norm_anchor[0])
            points = torch.cat([points_action[0].detach(), points_anchor[0].detach()], dim=0).cpu().numpy()
            goal_emb_on_objects = np.concatenate([
                points,
                np.concatenate([colors_action, colors_anchor], axis=0)],
                axis=-1)

            res_images['goal_emb_cond_x'] = wandb.Object3D(
                goal_emb_on_objects, markerSize=1000) #marker_scale * range_size)

            # Plot the p(z|Y) and p(z|X) goal embeddings as bar plots
            if self.plot_encoder_distribution:
                # Get embeddings for p(z|Y2)
                if 'points_action_aug_trans' in batch:
                    points_action_aug_trans = batch['points_action_aug_trans']
                    points_trans_action = batch['points_action_trans']
                    points_trans_anchor = batch['points_anchor_trans']
                    points_action = batch['points_action']
                    points_anchor = batch['points_anchor']
                    
                    model_no_cond_x_outputs = self.model(points_trans_action, 
                                                         points_trans_anchor, 
                                                         points_action_aug_trans, 
                                                         points_anchor, 
                                                         n_samples=self.n_samples)
                        
                    pzY2_emb = model_no_cond_x_outputs[0]['goal_emb']
                    
                    pzY2_action_emb = pzY2_emb[0, :, :points_action.shape[1]]
                    pzY2_anchor_emb = pzY2_emb[0, :, points_action.shape[1]:]
                    pzY2_action_dist = F.softmax(pzY2_action_emb, dim=-1).detach().cpu().numpy()[0]
                    pzY2_anchor_dist = F.softmax(pzY2_anchor_emb, dim=-1).detach().cpu().numpy()[0]
                    
                    pzY2_actions_probs = np.array([prob for prob in pzY2_action_dist])
                    pzY2_anchor_probs = np.array([prob for prob in pzY2_anchor_dist])
                
                pzY1_action_emb = goal_emb[0, :, :points_action.shape[1]]
                pzY1_anchor_emb = goal_emb[0, :, points_action.shape[1]:]
                pzY1_action_dist = F.softmax(pzY1_action_emb, dim=-1).detach().cpu().numpy()[0]
                pzY1_anchor_dist = F.softmax(pzY1_anchor_emb, dim=-1).detach().cpu().numpy()[0]
                
                pzX_action_dist = goal_emb_cond_x_norm_action.numpy()[0]
                pzX_anchor_dist = goal_emb_cond_x_norm_anchor.numpy()[0]
                
                x_vals = np.arange(pzY1_action_dist.shape[0])
                pzY1_actions_probs = np.array([prob for prob in pzY1_action_dist])
                pzY1_anchor_probs = np.array([prob for prob in pzY1_anchor_dist])

                pzX_actions_probs = np.array([prob for prob in pzX_action_dist])
                pzX_anchor_probs = np.array([prob for prob in pzX_anchor_dist])


                layout_margin = go.layout.Margin(
                    l=50, #left margin
                    r=120, #right margin
                    b=50, #bottom margin
                    t=50, #top margin
                    autoexpand=False 
                )

                # Plot action distributions
                action_max_prob = max(np.max(pzY1_actions_probs), np.max(pzX_actions_probs))
                
                pzY_action_data = [
                    go.Bar(name='pzY1', x=x_vals, y=pzY1_actions_probs, width=1, marker_color='blue', opacity=0.5, showlegend=True)
                ]
                if 'points_action_aug_trans' in batch:
                    pzY_action_data.append(
                        go.Bar(name='pzY2', x=x_vals, y=pzY2_actions_probs, width=1, marker_color='red', opacity=0.5, showlegend=True)
                    )
                    action_max_prob = max(action_max_prob, np.max(pzY2_actions_probs))
                
                pzY_action_plot = go.Figure(data=pzY_action_data)                
                pzY_action_plot.update_layout(barmode='overlay', 
                                              height=480, 
                                              width=1920, 
                                              yaxis_range=[0, action_max_prob*1.1], 
                                              margin=layout_margin,
                                              legend={'entrywidth': 40})
                
                pzX_action_plot = go.Figure(data=[
                    go.Bar(name='pzX', x=x_vals, y=pzX_actions_probs, width=1, marker_color='green', opacity=1, showlegend=True),
                ])
                pzX_action_plot.update_layout(barmode='overlay', 
                                              height=480, 
                                              width=1920, 
                                              yaxis_range=[0, action_max_prob*1.1], 
                                              margin=layout_margin,
                                              legend={'entrywidth': 40})
                
                
                # Plot anchor distributions
                anchor_max_prob = max(np.max(pzY1_anchor_probs), np.max(pzX_anchor_probs))
                
                pzY_anchor_data = [
                    go.Bar(name='pzY1', x=x_vals, y=pzY1_anchor_probs, width=1, marker_color='blue', opacity=0.5, showlegend=True)
                ]
                if 'points_action_aug_trans' in batch:
                    pzY_anchor_data.append(
                        go.Bar(name='pzY2', x=x_vals, y=pzY2_anchor_probs, width=1, marker_color='red', opacity=0.5, showlegend=True)
                    )
                    anchor_max_prob = max(anchor_max_prob, np.max(pzY2_anchor_probs))
                    
                pzY_anchor_plot = go.Figure(data=pzY_anchor_data)
                pzY_anchor_plot.update_layout(barmode='overlay', 
                                              height=480, 
                                              width=1920, 
                                              yaxis_range=[0, anchor_max_prob*1.1], 
                                              margin=layout_margin,
                                              legend={'entrywidth': 40})
                
                pzX_anchor_plot = go.Figure(data=[
                    go.Bar(name='pzX', x=x_vals, y=pzX_anchor_probs, width=1, marker_color='green', opacity=1, showlegend=True),
                ])
                pzX_anchor_plot.update_layout(barmode='overlay', 
                                              height=480, 
                                              width=1920, 
                                              yaxis_range=[0, anchor_max_prob*1.1], 
                                              margin=layout_margin,
                                              legend={'entrywidth': 40})
                
                res_images[log_prefix+'pzY_action_distribution'] = wandb.Html(plotly.io.to_html(pzY_action_plot, include_plotlyjs='cdn'))
                res_images[log_prefix+'pzX_action_distribution'] = wandb.Html(plotly.io.to_html(pzX_action_plot, include_plotlyjs='cdn'))
                res_images[log_prefix+'pzY_anchor_distribution'] = wandb.Html(plotly.io.to_html(pzY_anchor_plot, include_plotlyjs='cdn'))
                res_images[log_prefix+'pzX_anchor_distribution'] = wandb.Html(plotly.io.to_html(pzX_anchor_plot, include_plotlyjs='cdn'))

        return res_images
