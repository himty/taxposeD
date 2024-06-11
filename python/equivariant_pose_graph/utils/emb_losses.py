import numpy as np
from pytorch3d.transforms import Rotate, random_rotations
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from typing import Tuple, Dict

from equivariant_pose_graph.utils.occlusion_utils import (
    ball_occlusion, plane_occlusion
)

from plotly import graph_objects as go

mse_criterion = nn.MSELoss(reduction="sum")
to_tensor = ToTensor()

def dense_cos_similarity(psi, phi):
    phi = F.normalize(phi, dim=1)
    psi = F.normalize(psi, dim=1)
    return phi.transpose(-1, -2) @ psi

def neg_cos_sim(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z.detach(), dim=1)
    return -(p*z).sum(dim=1).mean()


def SimSiamLoss(model, pred_model, points, transform):
    points_centered = points - \
        points.mean(dim=1, keepdims=True)  # B, num_points, 3
    z = model(points_centered.transpose(1, 2))  # B, emb_dim, num_points
    p = pred_model(z)

    points_trans = transform.transform_points(points)
    points_trans_centered = points_trans - \
        points_trans.mean(dim=1, keepdims=True)  # B, num_points, 3

    z_trans = model(points_trans_centered.transpose(1, 2)).detach()
    p_trans = pred_model(z_trans)

    loss = neg_cos_sim(p, z_trans)/2 \
        + neg_cos_sim(p_trans, z)/2

    return loss, z, z_trans


def dist2mask(xyz, radius=0.02):
    d = (xyz.unsqueeze(1) - xyz.unsqueeze(2)).norm(dim=-1)
    w = (d > radius).float()
    w = w + \
        torch.eye(
            d.shape[-1], device=d.device).unsqueeze(0).tile([d.shape[0], 1, 1])
    return w


def dist2weight(xyz, func=None):
    d = (xyz.unsqueeze(1) - xyz.unsqueeze(2)).norm(dim=-1)
    if(func is not None):
        d = func(d)
    w = d / d.max(dim=-1, keepdims=True)[0]
    w = w + \
        torch.eye(
            d.shape[-1], device=d.device).unsqueeze(0).tile([d.shape[0], 1, 1])
    return w


def mean_order(similarity):
    order = (similarity > similarity.diagonal(
        dim1=-2, dim2=-1).unsqueeze(-1)).sum(-1)
    return order.float().mean() / similarity.shape[-1]


def mean_geo_diff(similarity, points):
    similarity_argmax = torch.argmax(similarity, dim=-1)  # B,num_points
    indices = similarity_argmax.unsqueeze(-1)  # B,num_points, 1
    indices = indices.repeat(1, 1, 3)  # B, num_points, 3
    most_similar_point = torch.gather(points, 1, indices)  # B,num_points, 3
    geo_diff = torch.norm(points - most_similar_point, dim=-1)

    return geo_diff.mean()