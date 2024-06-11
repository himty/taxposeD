import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import torch.nn.functional as F

def toDisplay(x, target_dim = None):
    while(target_dim is not None and x.dim() > target_dim):
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
                     mode='markers', marker=dict(size=2, color=colors[i],
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

empty_axis_dict = dict(
    backgroundcolor="rgba(0, 0, 0,0)",
    gridcolor="white",
    showbackground=True,
    zerolinecolor="white",
    showticklabels = False,
)
 
empty_background_dict = dict(
    xaxis = empty_axis_dict,
    yaxis = empty_axis_dict,
    zaxis = empty_axis_dict,
    xaxis_title='',
    yaxis_title='',
    zaxis_title='',
)

def flow_traces(
    pos, flows, sizeref=1.0, scene="scene", flowcolor="red", name="flow"
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]

    n_dest = n_pos + n_flows * sizeref

    for i in range(len(n_pos)):
        x_lines.append(n_pos[i][0])
        y_lines.append(n_pos[i][1])
        z_lines.append(n_pos[i][2])
        x_lines.append(n_dest[i][0])
        y_lines.append(n_dest[i][1])
        z_lines.append(n_dest[i][2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=10),
        name=name,
        hoverinfo = 'none',
    )

    head_trace = go.Scatter3d(
        x=n_dest[:, 0],
        y=n_dest[:, 1],
        z=n_dest[:, 2],
        mode="markers",
        marker={"size": 3, "color": "darkred"},
        scene=scene,
        showlegend=False,
        hoverinfo = 'none',
    )

    return [lines_trace, head_trace]

def visualize_correspondence(source_pts, target_pts, corr_scores, flow, 
        weights = None, skip=100):
    N_src = len(source_pts)
    N_tgt = len(target_pts)
    
    scatter_source = go.Scatter3d(
        x = source_pts[:,0], 
        y = source_pts[:,1], 
        z = source_pts[:,2], 
        mode = 'markers',
        hoverinfo = 'none',
    )

    scatter_target = go.Scatter3d(
        x = target_pts[:,0], 
        y = target_pts[:,1], 
        z = target_pts[:,2], 
        mode = 'markers',
        hoverinfo = 'none',
    )

#     scatter_target_base = go.Scatter3d(
#         x = target_pts[:,0], 
#         y = target_pts[:,1], 
#         z = target_pts[:,2], 
#         mode = 'markers',
#         hoverinfo = 'none',
#     )
    
    lines_flow, scatter_flow = flow_traces(source_pts[::skip], flow[::skip])
    lines_flow_selected, scatter_flow_selected = flow_traces(source_pts[:1], flow[:1])

    fig = go.FigureWidget(
        make_subplots(
            column_widths=[0.5, 0.25, 0.25],
            row_heights=[1],#[0.5, 0.5],
            rows=1, cols=3,
            specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
            # specs=[[{'type': 'surface', "rowspan": 2}, {'type': 'surface'}],[None, {'type': 'surface'}]],
        )
    )
    
    fig.add_trace(scatter_source, row=1,col=2)
    fig.add_trace(scatter_target, row=1,col=3)
    fig.add_trace(scatter_target, row=1,col=3)

    fig.add_trace(scatter_source, row=1,col=1)
    fig.add_trace(scatter_target, row=1,col=1)
    
    fig.add_trace(lines_flow, row=1,col=1)
    fig.add_trace(scatter_flow, row=1,col=1)

    fig.add_trace(lines_flow_selected, row=1,col=1)
    fig.add_trace(scatter_flow_selected, row=1,col=1)

    target_colors_joint = ['green'] * N_tgt
    source_colors_joint = ['#a3a7e4'] * N_src
    source_sizes_joint = [2] * N_src
    # source_sizes = [5] * N_src

    if(weights is not None):
        source_sizes = 20 * weights + 10
        weights_colors = weights
    else:
        source_sizes = [5] * N_src
        weights_colors = ['#a3a7e4'] * N_src

    scatter_source_joint = fig.data[3]
    scatter_source_joint.marker.size = source_sizes_joint
    # scatter_source_joint.marker.opacity = 0.5
    scatter_source_joint.marker.color = source_colors_joint
    scatter_source_joint.marker.line.width = 0

    scatter_target_joint = fig.data[4]
    scatter_target_joint.marker.size = [2] * N_tgt
    # scatter_target_joint.marker.opacity = 0.5
    scatter_target_joint.marker.color = target_colors_joint
    scatter_target_joint.marker.line.width = 0

    line_flow = fig.data[5]
    line_flow.line.color = 'rgba(255, 0, 0, 0.5)'
    line_flow.line.width = 1
    scatter_flow = fig.data[6]
    scatter_flow.marker.line.width = 0
    scatter_flow.marker.size = 2

    line_flow_selected = fig.data[7]
    scatter_flow_selected = fig.data[8]

    line_flow_selected.line.color = 'rgba(255, 0, 0, 0.5)'
    line_flow_selected.line.width = 0
    scatter_flow_selected.marker.line.width = 0
    scatter_flow_selected.marker.size = 0
    
    scatter_source = fig.data[0]
    scatter_target = fig.data[1]
    scatter_target_base = fig.data[2]

    scatter_source.marker.color = weights_colors
    scatter_source.marker.size = source_sizes
    scatter_source.marker.opacity = 0.5
    # scatter_source.marker.line.color = 0
    scatter_source.marker.line.width = 0

    scatter_target.marker.color = 'gray'
    scatter_target.marker.size = [5] * N_tgt
    scatter_target.marker.opacity = 0.0
    # scatter_target.marker.line.color = 0
    scatter_target.marker.line.width = 0

    scatter_target_base.marker.color = 'gray'
    scatter_target_base.marker.size = [5] * N_tgt
    scatter_target_base.marker.opacity = 0.1
    # scatter_target_base.marker.line.color = 0
    scatter_target_base.marker.line.width = 0
    # scatter_target_base.marker.size = 3

    fig.layout.hovermode = 'closest'
    
    def click_callback(trace, points, selector):
        c_src = weights_colors.copy()
        s_src = source_sizes.copy()
        for i in points.point_inds:
            # c_src[i] = 'red'
            s = corr_scores[i] / np.max(corr_scores[i])
            c_tgt = s
            s_tgt = s*20
            s_src[i] = 20
            
            flow_start = source_pts[i]
            flow_end = flow_start + flow[i]
            
            with fig.batch_update():
                scatter_source.marker.color = c_src
                scatter_target.marker.color = c_tgt
                # scatter_source_joint.marker.color = c_src
                scatter_source.marker.size = s_src
                scatter_target.marker.size = s_tgt
                # scatter_source_joint.marker.size = s_src
                
                line_flow_selected.x = (flow_start[0], flow_end[0], None)
                line_flow_selected.y = (flow_start[1], flow_end[1], None)
                line_flow_selected.z = (flow_start[2], flow_end[2], None)
                
                scatter_flow_selected.x = (flow_start[0], flow_end[0],)
                scatter_flow_selected.y = (flow_start[1], flow_end[1],)
                scatter_flow_selected.z = (flow_start[2], flow_end[2],)
                
                line_flow_selected.line.color = 'rgba(255, 0, 0, 0.5)'
                line_flow_selected.line.width = 5
                scatter_flow_selected.marker.line.width = 0
                scatter_flow_selected.marker.size = 5

    # def hover_callback(trace, points, selector):
    #     c = list(scatter.marker.color)
    #     s = list(scatter.marker.size)
    #     for i in points.point_inds:
    #         c[i] = 'red'
    #         with fig.batch_update():
    #             scatter.marker.color = c
    #             scatter.marker.size = s

    scatter_source.on_click(click_callback)
    # scatter_source.on_hover(click_callback)
    fig.update_layout(showlegend=False)
    fig.update_layout(
        scene = empty_background_dict,
        scene2 = empty_background_dict,
        scene3 = empty_background_dict,
    )
    return fig


def plot_taxposed_embeddings(points_action, points_anchor, ans, hydra_cfg):
    """
    Plot the TAXPoseD embeddings and selected points in 3D.
    
    Args:
        points_action (torch.Tensor): The action point cloud. 
        points_anchor (torch.Tensor): The anchor point cloud.
        ans (dict): The output of TAXPoseD inference. 
        hydra_cfg (omegaconf.DictConfig): The hydra config.
    """
    # assert hydra_cfg.model.return_debug and hydra_cfg.model_grasp.return_debug, "The model must return debug info."
        
    # Get the points for action, transformed action, and anchor
    points_action_data = np.array(points_action.squeeze(0).cpu())
    points_trans_action_data = np.array(ans["pred_points_action"].squeeze(0).cpu())
    points_anchor_data = np.array(points_anchor.squeeze(0).cpu())

    # Get the one-hot vectors for the selected action and anchor objects
    goal_emb_cond_x_norm_action_onehot = ans["trans_sample_action"].detach().cpu()
    goal_emb_cond_x_norm_anchor_onehot = ans["trans_sample_anchor"].detach().cpu()
    
    # Get the goal_emb_cond_x from p(z|X)
    goal_emb_cond_x = ans["goal_emb_cond_x"]
    
    # Get the distribution over the action points and the anchor points
    goal_emb_cond_x_norm_action = F.softmax(goal_emb_cond_x[0, :, :points_action.shape[1]], dim=-1).detach().cpu()
    goal_emb_cond_x_norm_anchor = F.softmax(goal_emb_cond_x[0, :, points_anchor.shape[1]:], dim=-1).detach().cpu()

    # Get the point selected by the one-hot vector for action, transformed action, and anchor
    action_selected_point = points_action_data[goal_emb_cond_x_norm_action_onehot[0].argmax()]
    trans_action_selected_point = points_trans_action_data[goal_emb_cond_x_norm_action_onehot[0].argmax()]
    anchor_selected_point = points_anchor_data[goal_emb_cond_x_norm_anchor_onehot[0].argmax()]

    # Create plotly traces for the action, transformed action, and anchor point clouds
    traces = []
    # Add the whole action point cloud
    traces.append(
        go.Scatter3d(
            mode="markers",
            marker={"size": 4, "color": "purple", "line": {"width": 0}},
            x=points_action_data[:, 0],
            y=points_action_data[:, 1],
            z=points_action_data[:, 2],
            name="action",
            scene="scene1"
        )
    )
    # Denote the selected action point
    traces.append(
        go.Scatter3d(
            mode="markers",
            marker={"size": 10, "color": "red", "line": {"width": 0}},
            x=[action_selected_point[0]],
            y=[action_selected_point[1]],
            z=[action_selected_point[2]],
            name="action selected point",
            scene="scene1"
        )
    )

    # Add the whole anchor point cloud
    traces.append(
        go.Scatter3d(
            mode="markers",
            marker={"size": 4, "color": goal_emb_cond_x_norm_anchor[0], "colorscale": "solar", "line": {"width": 0}},
            x=points_anchor_data[:, 0],
            y=points_anchor_data[:, 1],
            z=points_anchor_data[:, 2],
            name="anchor",
            scene="scene1"
        )
    )
    # Denote the selected anchor point
    traces.append(
        go.Scatter3d(
            mode="markers",
            marker={"size": 10, "color": "red", "line": {"width": 0}},
            x=[anchor_selected_point[0]],
            y=[anchor_selected_point[1]],
            z=[anchor_selected_point[2]],
            name="anchor selected point",
            scene="scene1"
        )
    )
    
    # Add the whole transformed action point cloud
    traces.append(
        go.Scatter3d(
            mode="markers",
            marker={"size": 4, "color": goal_emb_cond_x_norm_action[0], "colorscale": "viridis", "line": {"width": 0}},
            x=points_trans_action_data[:, 0],
            y=points_trans_action_data[:, 1],
            z=points_trans_action_data[:, 2],
            name="transformed action",
            scene="scene1"
        )
    )
    # Denote the selected transformed action point
    traces.append(
        go.Scatter3d(
            mode="markers",
            marker={"size": 10, "color": "red", "line": {"width": 0}},
            x=[trans_action_selected_point[0]],
            y=[trans_action_selected_point[1]],
            z=[trans_action_selected_point[2]],
            name="action selected point",
            scene="scene1"
        )
    )
    
    # Add traces to fig
    fig = go.Figure()
    fig.add_traces(traces)
    
    # Update layout following _3d_scene
    all_data = np.concatenate([points_action_data, points_anchor_data, points_trans_action_data], axis=0)
    all_data_mean = np.mean(all_data, axis=0)
    all_data_max_x = np.abs(all_data[:, 0] - all_data_mean[0]).max()
    all_data_max_y = np.abs(all_data[:, 1] - all_data_mean[1]).max()
    all_data_max_z = np.abs(all_data[:, 2] - all_data_mean[2]).max()
    all_max = max(all_data_max_x, all_data_max_y, all_data_max_z)
    scene1 = dict(
        xaxis=dict(nticks=10, range=[all_data_mean[0] - all_max, all_data_mean[0] + all_max]),
        yaxis=dict(nticks=10, range=[all_data_mean[1] - all_max, all_data_mean[1] + all_max]),
        zaxis=dict(nticks=10, range=[all_data_mean[2] - all_max, all_data_mean[2] + all_max]),
        aspectratio=dict(x=1, y=1, z=1),
    )
    fig.update_layout(
        scene1=scene1,
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=1.0, y=0.75)
    )

    # Display figure    
    fig.show()


def plot_all_predictions(points_action, points_anchor, ans_list, hydra_cfg):
    """
    Plot multiple TAXPoseD predictions in 3D.
    
    Args:
        points_action (torch.Tensor): The action point cloud. 
        points_anchor (torch.Tensor): The anchor point cloud.
        ans_list (dict): The output of TAXPoseD inference.
        hydra_cfg (omegaconf.DictConfig): The hydra config.
    """
    # assert hydra_cfg.model.return_debug and hydra_cfg.model_grasp.return_debug, "The model must return debug info."
        
    # Get the points for action, transformed action, and anchor
    points_action_data = points_action.detach().squeeze(0).cpu().numpy()
    points_anchor_data = points_anchor.detach().squeeze(0).cpu().numpy()

    # Create plotly traces for the action, transformed action, and anchor point clouds
    traces = []
    # Add the whole action point cloud
    traces.append(
        go.Scatter3d(
            mode="markers",
            marker={"size": 4, "color": "purple", "line": {"width": 0}},
            x=points_action_data[:, 0],
            y=points_action_data[:, 1],
            z=points_action_data[:, 2],
            name="action",
            scene="scene1"
        )
    )

    # Add the whole anchor point cloud
    traces.append(
        go.Scatter3d(
            mode="markers",
            marker={"size": 4, "color": "blue", "line": {"width": 0}},
            x=points_anchor_data[:, 0],
            y=points_anchor_data[:, 1],
            z=points_anchor_data[:, 2],
            name="anchor",
            scene="scene1"
        )
    )

    color_list = ["red", "green", "orange", "yellow", "cyan", "magenta", "pink", "brown", "gray", "black"]


    all_predictions_data = []
    for i, ans in enumerate(ans_list):

        points_trans_action_data = ans["pred_points_action"].detach().squeeze(0).cpu().numpy()
        all_predictions_data.append(points_trans_action_data)

        # Get the one-hot vectors for the selected action and anchor objects
        goal_emb_cond_x_norm_action_onehot = ans["trans_sample_action"].detach().cpu()
        goal_emb_cond_x_norm_anchor_onehot = ans["trans_sample_anchor"].detach().cpu()
        
        # # Get the goal_emb_cond_x from p(z|X)
        # goal_emb_cond_x = ans["goal_emb_cond_x"]
        
        # # Get the distribution over the action points and the anchor points
        # goal_emb_cond_x_norm_action = F.softmax(goal_emb_cond_x[0, :, :points_action.shape[1]], dim=-1).detach().cpu()
        # goal_emb_cond_x_norm_anchor = F.softmax(goal_emb_cond_x[0, :, points_anchor.shape[1]:], dim=-1).detach().cpu()

        # # Get the point selected by the one-hot vector for action, transformed action, and anchor
        # trans_action_selected_point = points_trans_action_data[goal_emb_cond_x_norm_action_onehot[0].argmax()]
        
        # Add the whole transformed action point cloud
        traces.append(
            go.Scatter3d(
                mode="markers",
                marker={"size": 4, "color": color_list[i], "line": {"width": 0}},
                x=points_trans_action_data[:, 0],
                y=points_trans_action_data[:, 1],
                z=points_trans_action_data[:, 2],
                name="transformed action",
                scene="scene1"
            )
        )
    
    # Add traces to fig
    fig = go.Figure()
    fig.add_traces(traces)
    
    # Update layout following _3d_scene
    all_data = np.concatenate([points_action_data, points_anchor_data, *all_predictions_data], axis=0)
    all_data_mean = np.mean(all_data, axis=0)
    all_data_max_x = np.abs(all_data[:, 0] - all_data_mean[0]).max()
    all_data_max_y = np.abs(all_data[:, 1] - all_data_mean[1]).max()
    all_data_max_z = np.abs(all_data[:, 2] - all_data_mean[2]).max()
    all_max = max(all_data_max_x, all_data_max_y, all_data_max_z)
    scene1 = dict(
        xaxis=dict(nticks=10, range=[all_data_mean[0] - all_max, all_data_mean[0] + all_max]),
        yaxis=dict(nticks=10, range=[all_data_mean[1] - all_max, all_data_mean[1] + all_max]),
        zaxis=dict(nticks=10, range=[all_data_mean[2] - all_max, all_data_mean[2] + all_max]),
        aspectratio=dict(x=1, y=1, z=1),
    )
    fig.update_layout(
        scene1=scene1,
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=1.0, y=0.75)
    )

    # Display figure    
    fig.show()
