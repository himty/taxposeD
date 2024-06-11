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


def load_data(filename, action_class, anchor_class):
    point_data = np.load(filename, allow_pickle=True)
    points_raw_np = point_data['clouds']
    classes_raw_np = point_data['classes']
        
    points_action_np = points_raw_np[classes_raw_np == action_class].copy()
    points_action_mean_np = points_action_np.mean(axis=0)
    points_action_np = points_action_np - points_action_mean_np

    points_anchor_np = points_raw_np[classes_raw_np == anchor_class].copy()
    points_anchor_np = points_anchor_np - points_action_mean_np
    points_anchor_mean_np = points_anchor_np.mean(axis=0)

    points_action = torch.from_numpy(points_action_np).float().unsqueeze(0)
    points_anchor = torch.from_numpy(points_anchor_np).float().unsqueeze(0)

    symmetric_cls = torch.Tensor([])
        
    return points_action, points_anchor, symmetric_cls


if __name__ == '__main__':
    dataset_dir = '/home/odonca/workspace/rpad/data/rpdiff/data/easy_utk4_10r_10m_100d/train'
    files = glob.glob(os.path.join(dataset_dir, '*.npz'))

    for i, file in enumerate(files):
        points_action, points_anchor, symmetric_cls = load_data(file, 0, 1)
        plot_multi_np([points_action[0].detach().cpu().numpy(), points_anchor[0].detach().cpu().numpy()])
        input("Press Enter to continue...")