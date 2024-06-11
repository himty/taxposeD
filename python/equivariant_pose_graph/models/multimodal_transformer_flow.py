#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP

import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from equivariant_pose_graph.models.pointnet2pyg import PN2DenseWrapper, PN2DenseParams, PN2EncoderWrapper, PN2EncoderParams

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(
        query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512, input_dims=3, num_heads=1, conditioning_size=0, last_relu=True):
        super(DGCNN, self).__init__()
        self.num_heads = num_heads
        self.conditioning_size = conditioning_size
        self.last_relu = last_relu

        self.conv1 = nn.Conv2d(input_dims*2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        
        if self.num_heads == 1:
            self.conv5 = nn.Conv2d(512 + self.conditioning_size, emb_dims, kernel_size=1, bias=False)
            self.bn5 = nn.BatchNorm2d(emb_dims)
        else:
            if self.conditioning_size > 0:
                raise NotImplementedError("Conditioning not implemented for multi-head DGCNN")
            self.conv5s = nn.ModuleList([nn.Conv2d(512, emb_dims, kernel_size=1, bias=False) for _ in range(self.num_heads)])
            self.bn5s = nn.ModuleList([nn.BatchNorm2d(emb_dims) for _ in range(self.num_heads)])

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        # bn5 defined above            

    def forward(self, x, conditioning=None):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        if self.conditioning_size == 0:
            assert conditioning is None
            x = torch.cat((x1, x2, x3, x4), dim=1)
        else:
            assert conditioning is not None
            x = torch.cat((x1, x2, x3, x4, conditioning[:,:,:,None]), dim=1)

        if self.num_heads == 1:
            x = self.bn5(self.conv5(x)).view(batch_size, -1, num_points)
        else:
            x = [bn5(conv5(x)).view(batch_size, -1, num_points) for bn5, conv5 in zip(self.bn5s, self.conv5s)]

        if self.last_relu:
            if self.num_heads == 1:
                x = F.relu(x)
            else:
                x = [F.relu(head) for head in x]
        return x

class DGCNNClassification(nn.Module):
    # Reference: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py#L88-L153

    def __init__(self, emb_dims=512, input_dims=3, num_heads=1, dropout=0.5, output_channels=40):
        super(DGCNNClassification, self).__init__()
        self.emb_dims = emb_dims
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.dropout=dropout
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(self.input_dims*2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, self.emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(self.emb_dims)

        self.linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)

        if self.num_heads == 1:
            self.linear3 = nn.Linear(256, self.output_channels)
        else:
            self.linear3s = nn.ModuleList([nn.Linear(256, self.output_channels) for _ in range(self.num_heads)])

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x).squeeze()
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)

        if self.num_heads == 1:
            x = self.linear3(x)[:,:,None]
        else:
            x = [linear3(x)[:,:,None] for linear3 in self.linear3s]

        return x

class Transformer(nn.Module):
    def __init__(self, emb_dims=512, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4, return_attn=False, bidirectional=True):
        super(Transformer, self).__init__()
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        self.return_attn = return_attn
        self.bidirectional = bidirectional
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(
                                        attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        src_embedding = self.model(
            tgt, src, None, None).transpose(2, 1).contiguous()
        src_attn = self.model.decoder.layers[-1].src_attn.attn

        if(self.bidirectional):
            tgt_embedding = self.model(
                src, tgt, None, None).transpose(2, 1).contiguous()
            tgt_attn = self.model.decoder.layers[-1].src_attn.attn

            if(self.return_attn):
                return src_embedding, tgt_embedding, src_attn, tgt_attn
            return src_embedding, tgt_embedding

        if(self.return_attn):
            return src_embedding, src_attn
        return src_embedding

class PointNet(nn.Module):
    def __init__(self, layer_dims=[3, 64, 64, 64, 128, 512]):
        super(PointNet, self).__init__()

        convs = []
        norms = []

        for j in range(len(layer_dims) - 1):
            convs.append(nn.Conv1d(
                layer_dims[j], layer_dims[j+1],
                kernel_size=1, bias=False))
            norms.append(nn.BatchNorm1d(layer_dims[j+1]))

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

    def forward(self, x):
        for bn, conv in zip(self.norms, self.convs):
            x = F.relu(bn(conv(x)))
        return x

class MLP(nn.Module):
    def __init__(self, emb_dims=512):
        super(MLP, self).__init__()
        self.input_fc = nn.Linear(emb_dims, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, emb_dims)

    def forward(self, x):

        # x = [batch size, emb_dims, num_points]
        batch_size, _, num_points = x.shape
        x = x.permute(0, -1, -2)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        h_1 = F.relu(self.input_fc(x))
        # batch size*num_points, 100
        h_2 = F.relu(self.hidden_fc(h_1))

        # batch size*num_points, output dim
        y_pred = self.output_fc(h_2)
        # batch size, num_points, output dim
        y_pred = y_pred.view(batch_size, num_points, -1)
        # batch size, emb_dims, num_points
        y_pred = y_pred.permute(0, 2, 1)

        return y_pred


class Multimodal_ResidualFlow_DiffEmbTransformer(nn.Module):
    EMB_DIMS_BY_CONDITIONING = {
        'pos_delta_l2norm': 1,
        'uniform_prior_pos_delta_l2norm': 1,
        'latent_z_linear_internalcond': 512,
    }

    # Number of heads that the DGCNN should output
    NUM_HEADS_BY_CONDITIONING = {
        'pos_delta_l2norm': 1,
        'uniform_prior_pos_delta_l2norm': 1,
        'latent_z_linear_internalcond': 2,
    }

    TP_INPUT_DIMS = {
        'pos_delta_l2norm': 3 + 1,
        'uniform_prior_pos_delta_l2norm': 3 + 1,
        'latent_z_linear_internalcond': 3,
    }

    def __init__(self, residualflow_diffembtransformer, gumbel_temp=0.5, freeze_residual_flow=False, freeze_z_embnn=False,
                 add_smooth_factor=0.05, conditioning="pos_delta_l2norm", latent_z_linear_size=40,
                 taxpose_centering="mean", pzY_input_dims=3, latent_z_cond_logvar_limit=0.0):
        super(Multimodal_ResidualFlow_DiffEmbTransformer, self).__init__()

        assert taxpose_centering in ["mean", "z"]
        assert conditioning in self.EMB_DIMS_BY_CONDITIONING.keys()

        self.latent_z_linear_size = latent_z_linear_size
        self.conditioning = conditioning
        self.taxpose_centering = taxpose_centering
        self.tax_pose = residualflow_diffembtransformer
        self.freeze_residual_flow = freeze_residual_flow
        self.center_feature = self.tax_pose.center_feature
        self.freeze_z_embnn = freeze_z_embnn
        self.freeze_embnn = self.tax_pose.freeze_embnn
        self.gumbel_temp = gumbel_temp
        self.add_smooth_factor = add_smooth_factor
        
        self.latent_z_cond_logvar_limit = latent_z_cond_logvar_limit

        # Embedding networks
        self.input_dims = pzY_input_dims
        self.emb_dims = self.EMB_DIMS_BY_CONDITIONING[self.conditioning]
        
        self.num_emb_heads = self.NUM_HEADS_BY_CONDITIONING[self.conditioning]


        # Point cloud with class labels between action and anchor
        if self.conditioning not in ["latent_z_linear_internalcond"]:
            # Single encoder
            print(f'--- P(z|Y) Using 1 PyG PN++ ---')
            args = PN2DenseParams()
            self.emb_nn_objs_at_goal = PN2DenseWrapper(in_channels=self.input_dims - 3, out_channels=self.emb_dims, p=args)
        else:
            print(f'--- P(z|Y) Using 1 PyG PN++ Classification ---')
            args = PN2EncoderParams()
            self.emb_nn_objs_at_goal = PN2EncoderWrapper(in_channels=self.input_dims - 3, out_channels=self.latent_z_linear_size, num_heads=self.num_emb_heads, emb_dims=self.emb_dims, p=args)

        # No transformer for p(z|Y)

    def get_dense_translation_point(self, points, ref, conditioning):
        """
            points- point cloud. (B, 3, num_points)
            ref- one hot vector (or nearly one-hot) that denotes the reference point
                     (B, num_points)

            Returns:
                dense point cloud. Each point contains the distance to the reference point (B, 3 or 1, num_points)
        """
        assert ref.ndim == 2
        assert torch.allclose(ref.sum(axis=1), torch.full((ref.shape[0], 1), 1, dtype=torch.float, device=ref.device))
        reference = (points*ref[:,None,:]).sum(axis=2)
        if conditioning in ["pos_delta_l2norm", "uniform_prior_pos_delta_l2norm", "distance_prior_pos_delta_l2norm", 
                            "pos_delta_l2norm_dist_vec", "uniform_prior_pos_delta_l2norm_dist_vec", "distance_prior_pos_delta_l2norm_dist_vec"]:
            dense = torch.norm(reference[:, :, None] - points, dim=1, keepdim=True)
        elif conditioning == "pos_exp_delta_l2norm":
            dense = torch.exp(-torch.norm(reference[:, :, None] - points, dim=1, keepdim=True)/1)
        elif conditioning == "pos_delta_vec":
            dense = reference[:, :, None] - points
        elif conditioning == "pos_loc3d":
            dense = reference[:,:,None].repeat(1, 1, 1024)
        elif conditioning == "pos_onehot":
            dense = ref[:, None, :]
        else:
            raise ValueError(f"Conditioning {conditioning} probably doesn't require a dense representation. This function is for" \
                                + "['pos_delta_l2norm', 'pos_delta_vec', 'pos_loc3d', 'pos_onehot', 'uniform_prior_pos_delta_l2norm']")
        return dense, reference
    
    
    def sample_dense_embedding(self, goal_emb, sampling_method='gumbel', n_samples=1):
        """Sample the dense goal embedding"""

        samples = []
        for i in range(n_samples):
            if sampling_method == 'gumbel':
                sample = F.gumbel_softmax(goal_emb, self.gumbel_temp, hard=True, dim=-1)
        
            elif sampling_method == 'random':
                rand_idx = torch.randint(0, goal_emb.shape[-1], (goal_emb.shape[0],))
                sample = torch.nn.functional.one_hot(rand_idx, num_classes=goal_emb.shape[-1]).float().to(goal_emb.device)
        
            elif sampling_method == 'top_n':
                top_idxs = torch.topk(goal_emb, n_samples, dim=-1)[1]
                sample = torch.nn.functional.one_hot(top_idxs[:, i], num_classes=goal_emb.shape[-1]).float().to(goal_emb.device)
        
            else:
                raise ValueError(f"Sampling method {sampling_method} not implemented")
        
            samples.append(sample)
        return samples
    
    
    def add_single_conditioning(self, goal_emb, points, conditioning, cond_type='action', sampling_method='gumbel', n_samples=1, z_samples=None):
        for_debug = {}

        sample_outputs = []
        if conditioning in ['pos_delta_l2norm', 'uniform_prior_pos_delta_l2norm']:

            goal_emb = goal_emb + self.add_smooth_factor

            # Only handle the translation case for now
            goal_emb_translation = goal_emb[:,0,:]

            translation_samples = Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(
                self, 
                goal_emb_translation, 
                sampling_method=sampling_method, 
                n_samples=n_samples
            )
            
            if z_samples is not None:
                translation_samples = z_samples[f"translation_samples_{cond_type}"]
                
            for translation_sample in translation_samples:
                # This is the only line that's different among the 3 different conditioning schemes in this category
                dense_trans_pt, ref = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                    None, 
                    points, 
                    translation_sample, 
                    conditioning=self.conditioning
                )

                points_and_cond = torch.cat([points] + [dense_trans_pt], axis=1)

                for_debug = {
                    f'dense_trans_pt_{cond_type}': dense_trans_pt,
                    f'trans_pt_{cond_type}': ref,
                    f'trans_sample_{cond_type}': translation_sample,
                    f'{cond_type}_points_and_cond': points_and_cond,
                }
                
                sample_outputs.append({
                    f'{cond_type}_points_and_cond': points_and_cond,
                    'for_debug': for_debug,
                })
        else:
            raise ValueError(f"Conditioning {conditioning} does not exist. Choose one of: {list(self.EMB_DIMS_BY_CONDITIONING.keys())}")

        return sample_outputs
    
    
    # TODO: rename to add_joint_conditioning, or merge the two functions
    def add_conditioning(self, goal_emb, action_points, anchor_points, conditioning, sampling_method='gumbel', n_samples=1, z_samples=None):
        for_debug = {}

        sample_outputs = []
        if conditioning in ['pos_delta_l2norm', 'uniform_prior_pos_delta_l2norm']:
            goal_emb = goal_emb + self.add_smooth_factor

            # Only handle the translation case for now
            goal_emb_translation = goal_emb[:,0,:]

            goal_emb_translation_action = goal_emb_translation[:, :action_points.shape[2]]
            goal_emb_translation_anchor = goal_emb_translation[:, action_points.shape[2]:]

            translation_samples_action = Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(self, goal_emb_translation_action, sampling_method=sampling_method, n_samples=n_samples)
            translation_samples_anchor = Multimodal_ResidualFlow_DiffEmbTransformer.sample_dense_embedding(self, goal_emb_translation_anchor, sampling_method=sampling_method, n_samples=n_samples)
            
            if z_samples is not None:
                translation_samples_action = z_samples["translation_samples_action"]
                translation_samples_anchor = z_samples["translation_samples_anchor"]
                
            for translation_sample_action, translation_sample_anchor in zip(translation_samples_action, translation_samples_anchor):
                # This is the only line that's different among the 3 different conditioning schemes in this category
                dense_trans_pt_action, ref_action = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(None, action_points, translation_sample_action, conditioning=conditioning)
                dense_trans_pt_anchor, ref_anchor = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(None, anchor_points, translation_sample_anchor, conditioning=conditioning)

                action_points_and_cond = torch.cat([action_points] + [dense_trans_pt_action], axis=1)
                anchor_points_and_cond = torch.cat([anchor_points] + [dense_trans_pt_anchor], axis=1)

                for_debug = {
                    'dense_trans_pt_action': dense_trans_pt_action,
                    'dense_trans_pt_anchor': dense_trans_pt_anchor,
                    'trans_pt_action': ref_action,
                    'trans_pt_anchor': ref_anchor,
                    'trans_sample_action': translation_sample_action,
                    'trans_sample_anchor': translation_sample_anchor,
                    'action_points_and_cond': action_points_and_cond,
                    'anchor_points_and_cond': anchor_points_and_cond,
                }
                
                sample_outputs.append({
                    'action_points_and_cond': action_points_and_cond,
                    'anchor_points_and_cond': anchor_points_and_cond,
                    'for_debug': for_debug,
                })
        elif conditioning in ["latent_z_linear_internalcond"]:
            # Do the reparametrization trick on the predicted mu and var
            def reparametrize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps * std + mu
            
            # Here, the goal emb has 2 heads. One for mean and one for variance
            goal_emb_mu = goal_emb[0]
            goal_emb_logvar = goal_emb[1]

            if self.latent_z_cond_logvar_limit > 0: # Hacky way to prevent large logvar causing NaNs
                goal_emb_logvar = self.latent_z_cond_logvar_limit * torch.tanh(goal_emb_logvar)
            
            goal_emb_sample = reparametrize(goal_emb_mu, goal_emb_logvar)

            for_debug = {
                'goal_emb_mu': goal_emb_mu,
                'goal_emb_logvar': goal_emb_logvar,
                'goal_emb_sample': goal_emb_sample
            }

            if conditioning == "latent_z_linear_internalcond":
                # The cond will be added in by TAXPose
                action_points_and_cond = action_points
                anchor_points_and_cond = anchor_points
                for_debug['goal_emb'] = goal_emb
            else:
                raise ValueError("Why is it here?")
            
            sample_outputs.append({
                'action_points_and_cond': action_points_and_cond,
                'anchor_points_and_cond': anchor_points_and_cond,
                'for_debug': for_debug,
            })
        else:
            raise ValueError(f"Conditioning {conditioning} does not exist. Choose one of: {list(self.EMB_DIMS_BY_CONDITIONING.keys())}")

        return sample_outputs

    def forward(self, *input, mode="forward", sampling_method="gumbel", n_samples=1, z_samples=None):
        # Forward pass goes through all of the model
        # Inference will use a sample from the prior if there is one
        #     - ex: conditioning = latent_z_linear_internalcond
        assert mode in ['forward', 'inference']

        action_points = input[0].permute(0, 2, 1)[:, :self.input_dims] # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :self.input_dims]

        if input[2] is None:
            mode = "inference"

        embedding_samples = []
        if mode == "forward":
            # Get the demonstration point clouds
            goal_action_points = input[2].permute(0, 2, 1)[:, :self.input_dims]
            goal_anchor_points = input[3].permute(0, 2, 1)[:, :self.input_dims]

            # Prepare the goal point clouds
            goal_action_points_dmean = goal_action_points
            goal_anchor_points_dmean = goal_anchor_points
            if self.center_feature:
                mean_goal = torch.cat([goal_action_points[:, :3], goal_anchor_points[:, :3]], axis=-1).mean(dim=2, keepdim=True)
                goal_action_points_dmean = goal_action_points[:, :3] - \
                                    mean_goal
                goal_anchor_points_dmean = goal_anchor_points[:, :3] - \
                                    mean_goal
                                    
                goal_action_points_dmean = torch.cat([goal_action_points_dmean, goal_action_points[:, 3:]], axis=1)
                goal_anchor_points_dmean = torch.cat([goal_anchor_points_dmean, goal_anchor_points[:, 3:]], axis=1)


            # Get the action and anchor embeddings jointly
            # Concatenate the action and anchor points
            goal_points_dmean = torch.cat([goal_action_points_dmean, goal_anchor_points_dmean], axis=2)

            # Obtain a goal embedding
            with torch.set_grad_enabled(not self.freeze_z_embnn):
                # Get the joint action/anchor embeddings
                goal_emb = self.emb_nn_objs_at_goal(goal_points_dmean)

            additional_logging = {}
            if self.conditioning in ["pos_delta_l2norm_dist_vec", "uniform_prior_pos_delta_l2norm_dist_vec", "distance_prior_pos_delta_l2norm_dist_vec"]:
                dist_vec = goal_emb[:, 1:]
                goal_emb = goal_emb[:, :1]
                additional_logging['dist_vec'] = dist_vec

            # Return n samples of spatially conditioned action and anchor points
            embedding_samples = self.add_conditioning(
                goal_emb, 
                action_points[:, :3], # Use only XYZ
                anchor_points[:, :3], # Use only XYZ
                self.conditioning, 
                sampling_method=sampling_method, 
                n_samples=n_samples,
                z_samples=z_samples
            )
            
            # Keep track of additional logging
            for i in range(len(embedding_samples)):
                embedding_samples[i]['goal_emb'] = goal_emb
                embedding_samples[i]['for_debug'] = {**embedding_samples[i]['for_debug'], **additional_logging}

        elif mode == "inference":
            # Return n samples of spatially conditioned action and anchor points
            for i in range(n_samples):
                action_points_and_cond, anchor_points_and_cond, goal_emb, for_debug = self.sample(action_points[:, :3], anchor_points[:, :3])
                embedding_samples.append({
                    'action_points_and_cond': action_points_and_cond,
                    'anchor_points_and_cond': anchor_points_and_cond,
                    'goal_emb': goal_emb,
                    'for_debug': for_debug
                })
        
        else:
            raise ValueError(f"Unknown mode {mode}")

        # Do the TAXPose forward pass
        outputs = []
        for embedding_sample in embedding_samples:
            action_points_and_cond = embedding_sample['action_points_and_cond']
            anchor_points_and_cond = embedding_sample['anchor_points_and_cond']
            goal_emb = embedding_sample['goal_emb']
            for_debug = embedding_sample['for_debug']
            
            # Optionally prepare the internal TAXPose DGCNN conditioning
            tax_pose_conditioning_action = None
            tax_pose_conditioning_anchor = None
            if self.conditioning == "latent_z_linear_internalcond":
                tax_pose_conditioning_action = torch.tile(for_debug['goal_emb_sample'], (1, 1, action_points.shape[-1]))
                tax_pose_conditioning_anchor = torch.tile(for_debug['goal_emb_sample'], (1, 1, anchor_points.shape[-1]))

            if self.taxpose_centering == "mean":
                # Mean center the action and anchor points
                action_center = action_points[:, :3].mean(dim=2, keepdim=True)
                anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)
            elif self.taxpose_centering == "z":
                # Center about the selected discrete spatial grounding action/anchor point
                action_center = for_debug['trans_pt_action'][:,:,None]
                anchor_center = for_debug['trans_pt_anchor'][:,:,None]
            else:
                raise ValueError(f"Unknown self.taxpose_centering: {self.taxpose_centering}")

            # Decode spatially conditioned action and anchor points into flows to obtain the goal configuration
            with torch.set_grad_enabled(not self.freeze_residual_flow):
                flow_action = self.tax_pose(
                    action_points_and_cond.permute(0, 2, 1), 
                    anchor_points_and_cond.permute(0, 2, 1),
                    conditioning_action=tax_pose_conditioning_action,
                    conditioning_anchor=tax_pose_conditioning_anchor,
                    action_center=action_center,
                    anchor_center=anchor_center
                )


            ########## LOGGING ############
            # Change goal_emb here to be what is going to be logged. For the latent_z conditioning, we just log the mean
            if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"] and goal_emb is not None:
                goal_emb = goal_emb[0] # This is mu

            flow_action = {
                **flow_action, 
                'goal_emb': goal_emb,
                **for_debug,
            }

            outputs.append(flow_action)
        return outputs
    
    
    def sample_single(self, points, sample_type="action",**input_kwargs):
        if self.conditioning in ['uniform_prior_pos_delta_l2norm', 'uniform_prior_pos_delta_l2norm_dist_vec']:
            # sample from a uniform prior
            N, B = points.shape[-1], points.shape[0]
            translation_sample = F.one_hot(torch.randint(N, (B,)), N).float().cuda()

            dense_trans_pt, ref = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(
                None, 
                points, 
                translation_sample, 
                conditioning=self.conditioning
            )

            points_and_cond = torch.cat([points] + [dense_trans_pt], axis=1)

            goal_emb = None

            for_debug = {
                f'dense_trans_pt_{sample_type}': dense_trans_pt,
                f'trans_pt_{sample_type}': ref,
                f'trans_sample_{sample_type}': translation_sample,
            }
        else:
            raise ValueError(f"Sampling not supported for conditioning {self.conditioning}. Pick one of the latent_z_xxx conditionings")
        return points_and_cond, goal_emb, for_debug
    
    
    def sample(self, action_points, anchor_points, **input_kwargs):
        if self.conditioning in ["latent_z_linear", "latent_z_linear_internalcond"]:
            # Take a SINGLE sample z ~ N(0,1)
            for_debug = {}
            if self.conditioning == "latent_z_linear":
                goal_emb = torch.tile(torch.randn((action_points.shape[0], self.latent_z_linear_size, 1)).to(action_points.device), (1, 1, action_points.shape[-1]))
                action_points_and_cond = torch.cat([action_points, goal_emb], axis=1)
                anchor_points_and_cond = torch.cat([anchor_points, goal_emb], axis=1)
            elif self.conditioning == "latent_z_linear_internalcond":
                goal_emb_sample = torch.randn((action_points.shape[0], self.latent_z_linear_size, 1)).to(action_points.device)
                action_points_and_cond = action_points
                anchor_points_and_cond = anchor_points
                for_debug['goal_emb_sample'] = goal_emb_sample
                goal_emb = None
            else:
                raise ValueError("Why is it here?")
        elif self.conditioning in ['uniform_prior_pos_delta_l2norm', 'uniform_prior_pos_delta_l2norm_dist_vec']:
            # sample from a uniform prior
            N_action, N_anchor, B = action_points.shape[-1], anchor_points.shape[-1], action_points.shape[0]
            translation_sample_action = F.one_hot(torch.randint(N_action, (B,)), N_action).float().cuda()
            translation_sample_anchor = F.one_hot(torch.randint(N_anchor, (B,)), N_anchor).float().cuda()

            dense_trans_pt_action, ref_action = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(None, action_points, translation_sample_action, conditioning=self.conditioning)
            dense_trans_pt_anchor, ref_anchor = Multimodal_ResidualFlow_DiffEmbTransformer.get_dense_translation_point(None, anchor_points, translation_sample_anchor, conditioning=self.conditioning)

            action_points_and_cond = torch.cat([action_points] + [dense_trans_pt_action], axis=1)
            anchor_points_and_cond = torch.cat([anchor_points] + [dense_trans_pt_anchor], axis=1)

            goal_emb = None

            for_debug = {
                'dense_trans_pt_action': dense_trans_pt_action,
                'dense_trans_pt_anchor': dense_trans_pt_anchor,
                'trans_pt_action': ref_action,
                'trans_pt_anchor': ref_anchor,
                'trans_sample_action': translation_sample_action,
                'trans_sample_anchor': translation_sample_anchor,
            }
        else:
            raise ValueError(f"Sampling not supported for conditioning {self.conditioning}. Pick one of the latent_z_xxx conditionings")
        return action_points_and_cond, anchor_points_and_cond, goal_emb, for_debug


class Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX(nn.Module):
    def __init__(self, residualflow_embnn, sample_z=True, 
                 return_debug=False,
                 pzX_transformer_embnn_dims=512, pzX_transformer_emb_dims=512,
                 pzX_input_dims=3
        ):
        super(Multimodal_ResidualFlow_DiffEmbTransformer_WithPZCondX, self).__init__()
        self.residflow_embnn = residualflow_embnn

        # Use the other class definition so that it matches between classes
        self.conditioning = self.residflow_embnn.conditioning
        self.num_emb_heads = self.residflow_embnn.num_emb_heads
        self.emb_dims = self.residflow_embnn.emb_dims
        self.input_dims = pzX_input_dims
        self.taxpose_centering = self.residflow_embnn.taxpose_centering
        self.freeze_residual_flow = self.residflow_embnn.freeze_residual_flow
        self.freeze_z_embnn = self.residflow_embnn.freeze_z_embnn
        self.freeze_embnn = self.residflow_embnn.freeze_embnn

        self.return_debug = return_debug

        self.add_smooth_factor = self.residflow_embnn.add_smooth_factor
        self.gumbel_temp = self.residflow_embnn.gumbel_temp
        self.sample_z = sample_z
        self.center_feature = self.residflow_embnn.center_feature

        self.latent_z_linear_size = self.residflow_embnn.latent_z_linear_size
        self.latent_z_cond_logvar_limit = self.residflow_embnn.latent_z_cond_logvar_limit
        
        # Embedding networks
        if self.conditioning not in ["latent_z_linear_internalcond"]:
            print(f'--- P(z|X) Using 2 DGCNN ---')
            self.p_z_cond_x_embnn_action = DGCNN(input_dims=self.input_dims, emb_dims=self.emb_dims, num_heads=self.num_emb_heads, last_relu=False)
            self.p_z_cond_x_embnn_anchor = DGCNN(input_dims=self.input_dims, emb_dims=self.emb_dims, num_heads=self.num_emb_heads, last_relu=False)
        else:
            self.emb_dims = self.latent_z_linear_size * 2
            print(f'--- P(z|X) Using 2 DGCNN Classification ---')
            self.p_z_cond_x_embnn_action = DGCNNClassification(input_dims=self.input_dims, emb_dims=512, num_heads=self.num_emb_heads, dropout=0.5, output_channels=self.latent_z_linear_size)
            self.p_z_cond_x_embnn_anchor = DGCNNClassification(input_dims=self.input_dims, emb_dims=512, num_heads=self.num_emb_heads, dropout=0.5, output_channels=self.latent_z_linear_size)

        # Set up the transformer
        self.pzX_transformer_embnn_dims = pzX_transformer_embnn_dims
        self.pzX_transformer_emb_dims = pzX_transformer_emb_dims

        print(f'--- P(z|X) Using Cross Object Transformer ---')
        if self.conditioning not in ["latent_z_linear_internalcond"]:
            print(f'------ With 2 DGCNN Encoders ------')
            self.p_z_cond_x_embnn_action = DGCNN(input_dims=self.input_dims, emb_dims=self.pzX_transformer_embnn_dims, num_heads=1, last_relu=False)
            self.p_z_cond_x_embnn_anchor = DGCNN(input_dims=self.input_dims, emb_dims=self.pzX_transformer_embnn_dims, num_heads=1, last_relu=False)
        
        self.p_z_cond_x_action_transformer = Transformer(emb_dims=self.pzX_transformer_emb_dims, return_attn=True, bidirectional=False)
        self.p_z_cond_x_anchor_transformer = Transformer(emb_dims=self.pzX_transformer_emb_dims, return_attn=True, bidirectional=False)

        self.action_proj = nn.Sequential(
            PointNet([self.pzX_transformer_emb_dims, 64, 64, 64, 128, 512]),
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
        )
        self.anchor_proj = nn.Sequential(
            PointNet([self.pzX_transformer_emb_dims, 64, 64, 64, 128, 512]),
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
        )

    def forward(self, *input, sampling_method="gumbel", n_samples=1, z_samples=None):
        action_points = input[0].permute(0, 2, 1)[:, :self.input_dims] # B,3,num_points
        anchor_points = input[1].permute(0, 2, 1)[:, :self.input_dims]

        # Prepare the action/anchor point clouds
        action_points_dmean = action_points
        anchor_points_dmean = anchor_points
        if self.residflow_embnn.center_feature:
            action_points_dmean = action_points[:, :3] - \
                action_points[:, :3].mean(dim=2, keepdim=True)
            anchor_points_dmean = anchor_points[:, :3] - \
                anchor_points[:, :3].mean(dim=2, keepdim=True)
                
            action_points_dmean = torch.cat([action_points_dmean, action_points[:, 3:]], axis=1)
            anchor_points_dmean = torch.cat([anchor_points_dmean, anchor_points[:, 3:]], axis=1)

        # Jointly predict the action and anchor goal embeddings
        # Obtain the goal embedding
        # Separately predict the action and anchor embeddings
        goal_emb_cond_x_action = self.p_z_cond_x_embnn_action(action_points_dmean)
        goal_emb_cond_x_anchor = self.p_z_cond_x_embnn_anchor(anchor_points_dmean)
        
        # Apply cross-object transformer
        if self.conditioning in ["latent_z_linear_internalcond"]:
            # These vectors contain [mu, logvar], so we should merge them for the transformer step
            mu_size = goal_emb_cond_x_action[0].shape[1]
            goal_emb_cond_x_action = torch.cat(goal_emb_cond_x_action, dim=1)
            goal_emb_cond_x_anchor = torch.cat(goal_emb_cond_x_anchor, dim=1)

        action_emb_tf, action_attn = self.p_z_cond_x_action_transformer(goal_emb_cond_x_action, goal_emb_cond_x_anchor)
        anchor_emb_tf, anchor_attn = self.p_z_cond_x_anchor_transformer(goal_emb_cond_x_anchor, goal_emb_cond_x_action)
        
        goal_emb_cond_x_action = self.action_proj(action_emb_tf)
        goal_emb_cond_x_anchor = self.anchor_proj(anchor_emb_tf)
            
        # Concatenate the action and anchor embeddings
        goal_emb_cond_x = torch.cat([goal_emb_cond_x_action, goal_emb_cond_x_anchor], dim=-1)

        if self.conditioning in ["latent_z_linear_internalcond"]:
            goal_emb_cond_x = [goal_emb_cond_x[:, :mu_size], goal_emb_cond_x[:, mu_size:]]

        
        # Get n samples of spatially conditioned action and anchor points
        embedding_samples = Multimodal_ResidualFlow_DiffEmbTransformer.add_conditioning(self, 
                                                                                        goal_emb_cond_x, 
                                                                                        action_points[:, :3], 
                                                                                        anchor_points[:, :3], 
                                                                                        self.conditioning, 
                                                                                        sampling_method=sampling_method, 
                                                                                        n_samples=n_samples,
                                                                                        z_samples=z_samples)
        
        # Do the TAXPose forward pass
        outputs = []
        for embedding_sample in embedding_samples:
            action_points_and_cond = embedding_sample['action_points_and_cond']
            anchor_points_and_cond = embedding_sample['anchor_points_and_cond']
            for_debug = embedding_sample['for_debug']
            
            # Optionally prepare the internal TAXPose DGCNN conditioning
            tax_pose_conditioning_action = None
            tax_pose_conditioning_anchor = None
            if self.conditioning == "latent_z_linear_internalcond":
                tax_pose_conditioning_action = torch.tile(for_debug['goal_emb_sample'][:,:,:1], (1, 1, action_points.shape[-1]))
                tax_pose_conditioning_anchor = torch.tile(for_debug['goal_emb_sample'][:,:,1:], (1, 1, anchor_points.shape[-1]))

            # Prepare TAXPose inputs
            if self.taxpose_centering == "mean":
                # Mean center the action and anchor points
                action_center = action_points[:, :3].mean(dim=2, keepdim=True)
                anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)
            elif self.taxpose_centering == "z":
                # Center about the selected discrete spatial grounding action/anchor point
                action_center = for_debug['trans_pt_action'][:,:,None]
                anchor_center = for_debug['trans_pt_anchor'][:,:,None]
            else:
                raise ValueError(f"Unknown self.taxpose_centering: {self.taxpose_centering}")


            # Decode spatially conditioned action and anchor points into flows to obtain the goal configuration
            flow_action = self.residflow_embnn.tax_pose(
                action_points_and_cond.permute(0, 2, 1), # Unpermute to match taxpose forward pass input 
                anchor_points_and_cond.permute(0, 2, 1), # Unpermute to match taxpose forward pass input
                conditioning_action=tax_pose_conditioning_action,
                conditioning_anchor=tax_pose_conditioning_anchor,
                action_center=action_center,
                anchor_center=anchor_center
            )


            # If the demo is available, get p(z|Y) goal embedding
            pzY_logging = {'goal_emb': None}
            if input[2] is not None:
                # Inputs 2 and 3 are the objects in demo positions
                # If we have access to these, we can run the pzY network
                pzY_results = self.residflow_embnn(*input, sampling_method=sampling_method, n_samples=1)
                pzY_logging['goal_emb'] = pzY_results[0]['goal_emb']
                
            flow_action = {
                **flow_action,
                'goal_emb_cond_x': goal_emb_cond_x,
                **pzY_logging,
                **for_debug,
            }
            
            outputs.append(flow_action)

        return outputs
