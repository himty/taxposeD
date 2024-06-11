#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pulled from DCP

import os
import sys
import copy
import math
from tkinter import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F

# Share a model with the other script
from equivariant_pose_graph.models.multimodal_transformer_flow import DGCNN

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

class ResidualMLPHead(nn.Module):
    """
    Base ResidualMLPHead with flow calculated as
    v_i = f(\phi_i) + \tilde{y}_i - x_i
    """

    def __init__(self, emb_dims=512, output_dims=3, pred_weight=True, residual_on=True, 
                 conditioning_type='old', conditioning_size=0):
        super(ResidualMLPHead, self).__init__()
        self.flow_emb_dims = emb_dims
        self.weight_emb_dims = emb_dims
        self.conditioning_type = conditioning_type
        
        self.emb_mul = 1
        
        # Add conditioning before flow weight predictions
        if self.conditioning_type in ['flow_fix-one_flow_head']:
            assert output_dims == 4, "Must predict 4 dimensions for flow and weight"

        if self.flow_emb_dims < 10:
            self.proj_flow = nn.Sequential(
                PointNet([self.flow_emb_dims*self.emb_mul, 64, 64, 64, 128, 512]),
                # PointNet([emb_dims, emb_dims//2, emb_dims//4, emb_dims//8]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )
        else:
            self.proj_flow = nn.Sequential(
                PointNet([self.flow_emb_dims*self.emb_mul, self.flow_emb_dims*self.emb_mul//2, self.flow_emb_dims*self.emb_mul//4, self.flow_emb_dims*self.emb_mul//8]),
                nn.Conv1d(self.flow_emb_dims*self.emb_mul//8, output_dims, kernel_size=1, bias=False),
            )
        self.pred_weight = pred_weight
        if self.pred_weight:
            self.proj_flow_weight = nn.Sequential(
                PointNet([self.weight_emb_dims, 64, 64, 64, 128, 512]),
                # PointNet([self.weight_emb_dims, self.weight_emb_dims//2, self.weight_emb_dims//4, self.weight_emb_dims//8]),
                nn.Conv1d(512, 1, kernel_size=1, bias=False),
            )

        self.residual_on = residual_on

    def forward(self, *input, scores=None, return_flow_component=False, return_embedding=False, conditioning=None):
        action_embedding = input[0]
        anchor_embedding = input[1]
        action_points = input[2]
        anchor_points = input[3]

        if(scores is None):
            if(len(input) <= 4):
                action_query = action_embedding
                anchor_key = anchor_embedding
            else:
                action_query = input[4]
                anchor_key = input[5]

            d_k = action_query.size(1)
            scores = torch.matmul(action_query.transpose(
                2, 1).contiguous(), anchor_key) / math.sqrt(d_k)
            # W_i # B, N, N (N=number of points, 1024 cur)
            scores = torch.softmax(scores, dim=2)

        corr_points = torch.matmul(
            anchor_points, scores.transpose(2, 1).contiguous())
        
        # \tilde{y}_i = sum_{j}{w_ij,y_j}, - x_i  # B, 3, N
        corr_flow = corr_points - action_points

        embedding = action_embedding  # B,512,N
        
        residual_flow = self.proj_flow(embedding)  # B,output_dims,N

        if self.conditioning_type in ['flow_fix-one_flow_head']:
            weight = residual_flow[:, 3:, :]
            residual_flow = residual_flow[:, :3, :]
            
        if self.residual_on:
            flow = residual_flow + corr_flow
        else:
            flow = corr_flow

        if self.pred_weight:
            if self.conditioning_type in ['flow_fix-one_flow_head']:
                # Use jointly predicted weight
                pass
            else:
                weight = self.proj_flow_weight(action_embedding)
            corr_flow_weight = torch.concat([flow, weight], dim=1)
        else:
            corr_flow_weight = flow
        return {
            'full_flow': corr_flow_weight,
            'residual_flow': residual_flow,
            'corr_flow': corr_flow,
            'corr_points': corr_points,
            'scores': scores,
        }

class ResidualFlow_DiffEmbTransformer(nn.Module):
    """
    This is the unimodal TAXPose model, slightly modified to accept conditionings
    """


    def __init__(self, emb_dims=512, input_dims=3, cycle=True, emb_nn='dgcnn', return_flow_component=False, center_feature=False,
                 pred_weight=True, residual_on=True, freeze_embnn=False, use_transformer_attention=True,
                 conditioning_size=0, sample=False,
                 conditioning_type='old', n_heads=4,
        ):
        super(ResidualFlow_DiffEmbTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.input_dims = input_dims
        self.cycle = cycle
        self.conditioning_size = conditioning_size
        self.center_feature = center_feature
        self.pred_weight = pred_weight
        self.residual_on = residual_on
        self.freeze_embnn = freeze_embnn
        self.use_transformer_attention = use_transformer_attention
        self.n_heads = n_heads

        self.conditioning_type = conditioning_type
        encoder_input_dims = self.input_dims
        head_output_dims = self.input_dims
        transformer_emb_dims = self.emb_dims
        head_emb_dims = self.emb_dims
        head_conditioning_size = 0
        # Use original conditioning with weird conditioning value flow as flow weight
        if self.conditioning_type == 'old':
            pass
        # Predict flow and weight in one head
        elif self.conditioning_type in ['flow_fix-one_flow_head']:
            encoder_input_dims = self.input_dims
            head_output_dims = 4
        else:
            raise ValueError(f"Invalid conditioning type {self.conditioning_type}")

        if emb_nn == 'dgcnn':
            print(f'--- TAXPose Decoder using 2 DGCNN ---')
            self.emb_nn_action = DGCNN(
                emb_dims=self.emb_dims,
                input_dims=encoder_input_dims, 
                conditioning_size=self.conditioning_size
            )
            self.emb_nn_anchor = DGCNN(
                emb_dims=self.emb_dims, 
                input_dims=encoder_input_dims, 
                conditioning_size=self.conditioning_size
            )
        else:
            raise Exception('Not implemented')

        self.transformer_action = Transformer(
            emb_dims=transformer_emb_dims, 
            return_attn=True, 
            bidirectional=False, 
            n_heads=self.n_heads
        )
        self.transformer_anchor = Transformer(
            emb_dims=transformer_emb_dims, 
            return_attn=True, 
            bidirectional=False, 
            n_heads=self.n_heads
        )
        
        self.head_action = ResidualMLPHead(
            emb_dims=head_emb_dims, 
            output_dims=head_output_dims, 
            pred_weight=self.pred_weight, 
            residual_on=self.residual_on, 
            conditioning_type=self.conditioning_type,
            conditioning_size=head_conditioning_size,
        )
        self.head_anchor = ResidualMLPHead(
            emb_dims=head_emb_dims, 
            output_dims=head_output_dims,
            pred_weight=self.pred_weight, 
            residual_on=self.residual_on,
            conditioning_type=self.conditioning_type,
            conditioning_size=head_conditioning_size,
        )
            
    def forward(self, *input, conditioning_action=None, conditioning_anchor=None, action_center=None, anchor_center=None):
        action_points = input[0].permute(0, 2, 1) # B,self.input_dims,num_points
        anchor_points = input[1].permute(0, 2, 1) # B,self.input_dims,num_points

        # TAX-Pose defaults
        if action_center is None:
            action_center = action_points[:, :3].mean(dim=2, keepdim=True)
        if anchor_center is None:
            anchor_center = anchor_points[:, :3].mean(dim=2, keepdim=True)

        action_points_dmean = torch.cat(
            [
                action_points[:,:3,:] - \
                    action_center,
                action_points[:,3:,:],
            ],
            dim=1
        )
        anchor_points_dmean = torch.cat(
            [
                anchor_points[:,:3,:] - \
                    anchor_center,
                anchor_points[:,3:,:],
            ],
            dim=1
        )
        # mean center point cloud before DGCNN
        if not self.center_feature:
            action_points_dmean = action_points
            anchor_points_dmean = anchor_points
            
        action_embedding = self.emb_nn_action(action_points_dmean, conditioning=conditioning_action)
        anchor_embedding = self.emb_nn_anchor(anchor_points_dmean, conditioning=conditioning_anchor)
        
        if self.freeze_embnn:
            action_embedding = action_embedding.detach()
            anchor_embedding = anchor_embedding.detach()
        
        # tilde_phi, phi are both B,512,N
        action_embedding_tf, action_attn = \
            self.transformer_action(action_embedding, anchor_embedding)
        anchor_embedding_tf, anchor_attn = \
            self.transformer_anchor(anchor_embedding, action_embedding)
            
        action_embedding_tf = action_embedding + action_embedding_tf
        anchor_embedding_tf = anchor_embedding + anchor_embedding_tf
        
        if(not self.use_transformer_attention):
            action_attn = None
            anchor_attn = None
        action_attn = action_attn.mean(dim=1)

        head_action_conditioning = None
        head_anchor_conditioning = None

        if self.conditioning_type.split('-')[0] == 'flow_fix':
            # Only pass in the XYZ points to the head network
            action_points = action_points[:, :3, :]
            anchor_points = anchor_points[:, :3, :]

        outputs = {}

        ### action2anchor ###
        flow_output_action = self.head_action(
            action_embedding_tf, 
            anchor_embedding_tf,
            action_points, 
            anchor_points, 
            scores=action_attn, 
            conditioning=head_action_conditioning
        )
        flow_action = flow_output_action['full_flow'].permute(0, 2, 1)
        residual_flow_action = flow_output_action['residual_flow'].permute(0, 2, 1)
        corr_flow_action = flow_output_action['corr_flow'].permute(0, 2, 1)
        
        outputs["flow_action"] = flow_action
        outputs["residual_flow_action"] = residual_flow_action
        outputs["corr_flow_action"] = corr_flow_action
        outputs["corr_points_action"] = flow_output_action['corr_points']
        outputs["scores_action"] = flow_output_action['scores']
        
        if "P_A" in flow_output_action:
            original_points_action = flow_output_action["P_A"].permute(0, 2, 1)
            outputs["original_points_action"] = original_points_action
            outputs["sampled_ixs_action"] = flow_output_action["A_ixs"]


        ### anchor2action ###
        anchor_attn = anchor_attn.mean(dim=1)
        flow_output_anchor = self.head_anchor(
            anchor_embedding_tf, 
            action_embedding_tf, 
            anchor_points, 
            action_points, 
            scores=anchor_attn, 
            conditioning=head_anchor_conditioning
        )
        flow_anchor = flow_output_anchor['full_flow'].permute(0, 2, 1)
        residual_flow_anchor = flow_output_anchor['residual_flow'].permute(0, 2, 1)
        corr_flow_anchor = flow_output_anchor['corr_flow'].permute(0, 2, 1)
        
        outputs["flow_anchor"] = flow_anchor
        outputs["residual_flow_anchor"] = residual_flow_anchor
        outputs["corr_flow_anchor"] = corr_flow_anchor
        outputs["corr_points_anchor"] = flow_output_anchor['corr_points']
        outputs["scores_anchor"] = flow_output_anchor['scores']
        
        if "P_A" in flow_output_anchor:
            original_points_anchor = flow_output_anchor["P_A"].permute(0, 2, 1)
            outputs["original_points_anchor"] = original_points_anchor
            outputs["sampled_ixs_anchor"] = flow_output_anchor["A_ixs"]
            
                
        return outputs


class FlowDecoder(nn.Module):
    def __init__(self, freeze_embnn=False):
        super(FlowDecoder, self).__init__()
        
        self.freeze_embnn = freeze_embnn
        self.emb_dims = 512
        self.input_dims = 4
        self.output_dims = 4 # flow + weight
        
        self.emb_nn_action = DGCNN(
            emb_dims=self.emb_dims,
            input_dims=self.input_dims, 
            conditioning_size=0
        )
        self.emb_nn_anchor = DGCNN(
            emb_dims=self.emb_dims,
            input_dims=self.input_dims,  
            conditioning_size=0
        )
        
        self.transformer_action = Transformer(
            emb_dims=self.emb_dims, 
            return_attn=True, 
            bidirectional=False, 
            n_heads=4
        )
        
        self.flow_head = nn.Sequential(
            PointNet([self.emb_dims, self.emb_dims//2, self.emb_dims//4, self.emb_dims//8]),
            nn.Conv1d(self.emb_dims//8, self.output_dims, kernel_size=1, bias=False),
        )
        
        print('Flow Decoder\n'*5)
        
    def forward(self, *input, conditioning_action=None, conditioning_anchor=None, action_center=None, anchor_center=None):
        print(f'Flow Decoder')
        
        action_points = input[0].permute(0, 2, 1) # B,self.input_dims,num_points
        anchor_points = input[1].permute(0, 2, 1) # B,self.input_dims,num_points

        assert action_center is not None and anchor_center is not None, "Must provide action and anchor center"

        # Center the point clouds about their respective centers
        action_points_dmean = torch.cat(
            [
                action_points[:,:3,:] - \
                    action_center,
                action_points[:,3:,:],
            ],
            dim=1
        )
        anchor_points_dmean = torch.cat(
            [
                anchor_points[:,:3,:] - \
                    anchor_center,
                anchor_points[:,3:,:],
            ],
            dim=1
        )
        
        # Get the embeddings for the action and anchor point clouds
        action_embedding = self.emb_nn_action(action_points_dmean, conditioning=conditioning_action)
        anchor_embedding = self.emb_nn_anchor(anchor_points_dmean, conditioning=conditioning_anchor)
        
        # Get the transformer embeddings
        action_embedding_tf, action_attn = self.transformer_action(action_embedding, anchor_embedding)
        action_embedding = action_embedding + action_embedding_tf
        
        # Get the flow and weights
        flow_and_weights = self.flow_head(action_embedding)
                
        return {
            'flow_action': flow_and_weights.permute(0, 2, 1),
        }        
        