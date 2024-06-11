import torch

from equivariant_pose_graph.utils.se3 import (
    get_degree_angle, 
    get_translation, 
    get_2transform_min_rotation_errors, 
    get_2transform_min_translation_errors,
    get_transform_list_min_rotation_errors,
    get_transform_list_min_translation_errors
)
from equivariant_pose_graph.utils.visualizations import plot_multi_np

def get_2rack_errors(pred_T_action, T0, T1, mode="batch_min_rack", verbose=False, T_aug_list=None, action_center=None, anchor_center=None):
    """
    Debugging function for calculating the error in predicting a mug-on-rack transform
    when there are 2 possible racks to place the mug on, and they are 0.3m apart in the x direction
    """
    assert mode in ["demo_rack", "bad_min_rack", "batch_min_rack", "aug_min_rack"]

    if mode == "demo_rack":
        error_R_max, error_R_min, error_R_mean = get_degree_angle(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))

        error_t_max, error_t_min, error_t_mean = get_translation(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))
    elif mode == "bad_min_rack":
        error_R_max0, error_R_min0, error_R_mean0 = get_degree_angle(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))
        error_t_max0, error_t_min0, error_t_mean0 = get_translation(T0.inverse().compose(
            T1).compose(pred_T_action.inverse()))
        error_R_max1, error_R_min1, error_R_mean1 = get_degree_angle(T0.inverse().translate(0.3, 0, 0).compose(
            T1).compose(pred_T_action.inverse()))
        error_R_max2, error_R_min2, error_R_mean2 = get_degree_angle(T0.inverse().translate(-0.3, 0, 0).compose(
            T1).compose(pred_T_action.inverse()))
        error_t_max1, error_t_min1, error_t_mean1 = get_translation(T0.inverse().translate(0.3, 0, 0).compose(
            T1.compose(pred_T_action.inverse())))
        error_t_max2, error_t_min2, error_t_mean2 = get_translation(T0.inverse().translate(-0.3, 0, 0).compose(
            T1.compose(pred_T_action.inverse())))
        error_R_mean = min(error_R_mean0, error_R_mean1, error_R_mean2)
        error_t_mean = min(error_t_mean0, error_t_mean1, error_t_mean2)
        if verbose:
            print(f'\t\/ a min over {error_t_mean0:.3f}, {error_t_mean1:.3f}, {error_t_mean2:.3f}')
    elif mode == "batch_min_rack":
        T = T0.inverse().compose(T1).compose(pred_T_action.inverse())
        Ts = torch.stack([
            T0.inverse().compose(T1).compose(pred_T_action.inverse()).get_matrix(),
            T0.inverse().translate(0.3, 0, 0).compose(T1).compose(pred_T_action.inverse()).get_matrix(),
            T0.inverse().translate(-0.3, 0, 0).compose(T1).compose(pred_T_action.inverse()).get_matrix(),
        ])

        error_R_mean, error_t_mean = 0, 0
        B = T0.get_matrix().shape[0]
        if verbose:
            print('\t\/ an average over ', end="")
        for b in range(B):  # for every batch
            _max, error_R_min, _mean = get_degree_angle(Transform3d(matrix=Ts[:, b, :, :]))
            error_R_mean += error_R_min

            _max, error_t_min, _mean = get_translation(Transform3d(matrix=Ts[:, b, :, :]))
            error_t_mean += error_t_min
            if verbose:
                print(f"{error_t_min:.3f}", end=" ")
        if verbose:
            print()
        error_R_mean /= B
        error_t_mean /= B
    elif mode == "aug_min_rack":
        assert T_aug_list is not None, "T_aug_list must be provided for aug_min_rack mode"
        
        gt_T_action = T0.inverse().compose(T1)
        if action_center is not None and anchor_center is not None:
            gt_T_action = action_center.inverse().compose(gt_T_action).compose(anchor_center)
        
        aug_T_list = []
        for T_aug in T_aug_list:
            aug_T_action = T0.inverse().compose(T_aug).compose(T1)
            if action_center is not None and anchor_center is not None:
                aug_T_action = action_center.inverse().compose(aug_T_action).compose(anchor_center)
            aug_T_list.append(aug_T_action)
        
        T_demo = gt_T_action.compose(pred_T_action.inverse())
        
        T_distractor_list = []
        for aug_T_action in aug_T_list:
            T_distractor = aug_T_action.compose(pred_T_action.inverse())
            T_distractor_list.append(T_distractor)

        error_t_max, error_t_min, error_t_mean = get_transform_list_min_translation_errors(T_demo, T_distractor_list)
        error_R_max, error_R_min, error_R_mean = get_transform_list_min_rotation_errors(T_demo, T_distractor_list)
    else:
        raise ValueError("Invalid rack error type!")

    return error_R_mean, error_t_mean


def print_rack_errors(name, error_R_mean, error_t_mean):
    print(f"{name}- R error: {error_R_mean:.3f}, t error: {error_t_mean:.3f}")
    
    
def get_all_sample_errors(pred_T_actions, T0, T1, mode="demo_rack", T_aug_list=None):
    error_R_maxs, error_R_mins, error_R_means = [], [], []
    error_t_maxs, error_t_mins, error_t_means = [], [], []
    for pred_T_action in pred_T_actions:
        if mode == "demo_rack":
            error_R_max, error_R_min, error_R_mean = get_degree_angle(T0.inverse().compose(
                    T1).compose(pred_T_action.inverse()))

            error_t_max, error_t_min, error_t_mean = get_translation(T0.inverse().compose(
                T1).compose(pred_T_action.inverse()))
        elif mode == "aug_min_rack":
            assert T_aug_list is not None, "T_aug_list must be provided for aug_min_rack mode"
        
            gt_T_action = T0.inverse().compose(T1)
            
            aug_T_list = []
            for T_aug in T_aug_list:
                aug_T_action = T0.inverse().compose(T_aug).compose(T1)
                aug_T_list.append(aug_T_action)
            
            T_demo = gt_T_action.compose(pred_T_action.inverse())
            
            T_distractor_list = []
            for aug_T_action in aug_T_list:
                T_distractor = aug_T_action.compose(pred_T_action.inverse())
                T_distractor_list.append(T_distractor)

            error_t_max, error_t_min, error_t_mean = get_transform_list_min_translation_errors(T_demo, T_distractor_list)
            error_R_max, error_R_min, error_R_mean = get_transform_list_min_rotation_errors(T_demo, T_distractor_list)
        else:
            raise ValueError(f"Sample errors not implemented for: {mode}") 
            
        error_R_maxs.append(error_R_max)
        error_R_mins.append(error_R_min)
        error_R_means.append(error_R_mean)
        error_t_maxs.append(error_t_max)
        error_t_mins.append(error_t_min)
        error_t_means.append(error_t_mean)
        
    return error_R_maxs, error_R_mins, error_R_means, error_t_maxs, error_t_mins, error_t_means