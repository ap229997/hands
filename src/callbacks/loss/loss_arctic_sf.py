import torch
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix

from src.utils.loss_modules import (
    compute_contact_devi_loss,
    hand_kp3d_loss,
    joints_loss,
    mano_loss,
    object_kp3d_loss,
    vector_loss,
    grasp_loss,
    render_loss,
)

l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")


def compute_loss_light(pred, gt, meta_info, args):
    # unpacking pred and gt
    pred_betas_r = pred["mano.beta.r"]
    pred_rotmat_r = pred["mano.pose.r"]
    pred_joints_r = pred["mano.j3d.cam.r"]
    pred_projected_keypoints_2d_r = pred["mano.j2d.norm.r"]
    pred_betas_l = pred["mano.beta.l"]
    pred_rotmat_l = pred["mano.pose.l"]
    pred_joints_l = pred["mano.j3d.cam.l"]
    pred_projected_keypoints_2d_l = pred["mano.j2d.norm.l"]

    gt_pose_r = gt["mano.pose.r"]
    gt_betas_r = gt["mano.beta.r"]
    gt_joints_r = gt["mano.j3d.cam.r"]
    gt_keypoints_2d_r = gt["mano.j2d.norm.r"]
    gt_pose_l = gt["mano.pose.l"]
    gt_betas_l = gt["mano.beta.l"]
    gt_joints_l = gt["mano.j3d.cam.l"]
    gt_keypoints_2d_l = gt["mano.j2d.norm.l"]

    is_valid = gt["is_valid"]
    right_valid = gt["right_valid"]
    left_valid = gt["left_valid"]
    joints_valid_r = gt["joints_valid_r"]
    joints_valid_l = gt["joints_valid_l"]

    # reshape
    gt_pose_r = axis_angle_to_matrix(gt_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    gt_pose_l = axis_angle_to_matrix(gt_pose_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    # Compute loss on MANO parameters
    loss_regr_pose_r, loss_regr_betas_r = mano_loss(
        pred_rotmat_r,
        pred_betas_r,
        gt_pose_r,
        gt_betas_r,
        criterion=mse_loss,
        is_valid=right_valid,
        return_mean=False,
    )
    loss_regr_pose_l, loss_regr_betas_l = mano_loss(
        pred_rotmat_l,
        pred_betas_l,
        gt_pose_l,
        gt_betas_l,
        criterion=mse_loss,
        is_valid=left_valid,
        return_mean=False,
    )

    # Compute 2D reprojection loss for the keypoints
    loss_keypoints_r = joints_loss(
        pred_projected_keypoints_2d_r,
        gt_keypoints_2d_r,
        criterion=mse_loss,
        jts_valid=joints_valid_r,
        return_mean=False,
    )
    loss_keypoints_l = joints_loss(
        pred_projected_keypoints_2d_l,
        gt_keypoints_2d_l,
        criterion=mse_loss,
        jts_valid=joints_valid_l,
        return_mean=False,
    )

    # Compute 3D keypoint loss
    loss_keypoints_3d_r = hand_kp3d_loss(
        pred_joints_r, gt_joints_r, mse_loss, joints_valid_r, return_mean=False,
    )
    loss_keypoints_3d_l = hand_kp3d_loss(
        pred_joints_l, gt_joints_l, mse_loss, joints_valid_l, return_mean=False,
    )

    loss_transl_l = vector_loss(
        pred["mano.cam_t.wp.l"] - pred["mano.cam_t.wp.r"],
        gt["mano.cam_t.wp.l"] - gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid * left_valid,
        return_mean=False,
    )

    loss_cam_t_r = vector_loss(
        pred["mano.cam_t.wp.r"],
        gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid,
        return_mean=False,
    )
    loss_cam_t_l = vector_loss(
        pred["mano.cam_t.wp.l"],
        gt["mano.cam_t.wp.l"],
        mse_loss,
        left_valid,
        return_mean=False,
    ) 

    loss_cam_t_r += vector_loss(
        pred["mano.cam_t.wp.init.r"],
        gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid,
        return_mean=False,
    )
    loss_cam_t_l += vector_loss(
        pred["mano.cam_t.wp.init.l"],
        gt["mano.cam_t.wp.l"],
        mse_loss,
        left_valid,
        return_mean=False,
    )

    bz = meta_info['is_j2d_loss'].shape[0]
    
    loss_cam_t_r = loss_cam_t_r.reshape(bz,-1) * meta_info['is_cam_loss'][..., None]
    loss_cam_t_l = loss_cam_t_l.reshape(bz,-1) * meta_info['is_cam_loss'][..., None]
    loss_keypoints_r = loss_keypoints_r.reshape(bz,-1) * meta_info['is_j2d_loss'][..., None]
    loss_keypoints_l = loss_keypoints_l.reshape(bz,-1) * meta_info['is_j2d_loss'][..., None]
    loss_keypoints_3d_r = loss_keypoints_3d_r.reshape(bz,-1) * meta_info['is_j3d_loss'][..., None]
    loss_keypoints_3d_l = loss_keypoints_3d_l.reshape(bz,-1) * meta_info['is_j3d_loss'][..., None]
    loss_regr_pose_r = loss_regr_pose_r.reshape(bz,-1) * meta_info['is_pose_loss'][..., None]
    loss_regr_pose_l = loss_regr_pose_l.reshape(bz,-1) * meta_info['is_pose_loss'][..., None]
    loss_regr_betas_r = loss_regr_betas_r.reshape(bz,-1) * meta_info['is_beta_loss'][..., None]
    loss_regr_betas_l = loss_regr_betas_l.reshape(bz,-1) * meta_info['is_beta_loss'][..., None]
    loss_transl_l = loss_transl_l.reshape(bz,-1) * meta_info['is_cam_loss'][..., None]

    loss_dict = {
        "loss/mano/cam_t/r": (loss_cam_t_r.mean().view(-1), 1.0),
        "loss/mano/cam_t/l": (loss_cam_t_l.mean().view(-1), 1.0),
        "loss/mano/kp2d/r": (loss_keypoints_r.mean().view(-1), 5.0),
        "loss/mano/kp3d/r": (loss_keypoints_3d_r.mean().view(-1), 5.0),
        "loss/mano/pose/r": (loss_regr_pose_r.mean().view(-1), 10.0),
        "loss/mano/beta/r": (loss_regr_betas_r.mean().view(-1), 0.001),
        "loss/mano/kp2d/l": (loss_keypoints_l.mean().view(-1), 5.0),
        "loss/mano/kp3d/l": (loss_keypoints_3d_l.mean().view(-1), 5.0),
        "loss/mano/pose/l": (loss_regr_pose_l.mean().view(-1), 10.0),
        "loss/mano/transl/l": (loss_transl_l.mean().view(-1), 1.0),
        "loss/mano/beta/l": (loss_regr_betas_l.mean().view(-1), 0.001),
    }

    if args.get('use_grasp_loss', False):
        gt_grasp_r = gt["grasp.r"]
        pred_grasp_r = pred["grasp.r"]
        gt_grasp_l = gt["grasp.l"]
        pred_grasp_l = pred["grasp.l"]
        loss_grasp_r = grasp_loss(pred_grasp_r, gt_grasp_r, gt['grasp_valid_r'], return_mean=False)
        loss_grasp_l = grasp_loss(pred_grasp_l, gt_grasp_l, gt['grasp_valid_l'], return_mean=False)
        loss_grasp_r = loss_grasp_r.reshape(bz,-1) * meta_info['is_grasp_loss'][..., None]
        loss_grasp_l = loss_grasp_l.reshape(bz,-1) * meta_info['is_grasp_loss'][..., None]
        loss_dict['loss/grasp/r'] = (loss_grasp_r.mean().view(-1), 0.1)
        loss_dict['loss/grasp/l'] = (loss_grasp_l.mean().view(-1), 0.1)

    if args.get('use_render_seg_loss', False):
        gt_mask_r = gt["render.r"]
        pred_mask_r = pred["render.r"]
        gt_mask_l = gt["render.l"]
        pred_mask_l = pred["render.l"]
        loss_mask_r = render_loss(pred_mask_r, gt_mask_r, gt['render_valid_r'], return_mean=False)
        loss_mask_l = render_loss(pred_mask_l, gt_mask_l, gt['render_valid_l'], return_mean=False)
        loss_mask_r = loss_mask_r.reshape(bz,-1) * meta_info['is_mask_loss'][..., None]
        loss_mask_l = loss_mask_l.reshape(bz,-1) * meta_info['is_mask_loss'][..., None]
        loss_dict['loss/mask/r'] = (loss_mask_r.mean().view(-1), 10.0)
        loss_dict['loss/mask/l'] = (loss_mask_l.mean().view(-1), 10.0)

    if args.get('use_depth_loss', False):
        gt_depth_r = gt['depth.r']
        pred_depth_r = pred['depth.r']
        gt_depth_l = gt['depth.l']
        pred_depth_l = pred['depth.l']
        loss_depth_r = l1_loss(pred_depth_r, gt_depth_r)
        loss_depth_l = l1_loss(pred_depth_l, gt_depth_l)
        loss_depth_r = loss_depth_r.reshape(bz,-1) * meta_info['is_depth_loss'][..., None]
        loss_depth_l = loss_depth_l.reshape(bz,-1) * meta_info['is_depth_loss'][..., None]
        loss_dict['loss/depth/r'] = (loss_depth_r.mean().view(-1), 1.0)
        loss_dict['loss/depth/l'] = (loss_depth_l.mean().view(-1), 1.0)

    if args.regress_center_corner:
        # center and corner regression loss
        loss_center_r = vector_loss(pred["center.r"], gt["center.r"], mse_loss, right_valid, return_mean=False)
        loss_center_l = vector_loss(pred["center.l"], gt["center.l"], mse_loss, left_valid, return_mean=False)
        loss_corner_r = vector_loss(pred["corner.r"], gt["corner.r"], mse_loss, right_valid, return_mean=False)
        loss_corner_l = vector_loss(pred["corner.l"], gt["corner.l"], mse_loss, left_valid, return_mean=False)
        loss_dict["loss/center/r"] = (loss_center_r.mean().view(-1), 1.0)
        loss_dict["loss/center/l"] = (loss_center_l.mean().view(-1), 1.0)
        loss_dict["loss/corner/r"] = (loss_corner_r.mean().view(-1), 1.0)
        loss_dict["loss/corner/l"] = (loss_corner_l.mean().view(-1), 1.0)
    return loss_dict
