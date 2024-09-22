import copy
import warnings

import numpy as np
import torch
from torch._C import is_anomaly_check_nan_enabled

import common.metrics as metrics
from common.torch_utils import unpad_vtensor

warnings.filterwarnings("ignore")

import torch

import common.torch_utils as torch_utils
from common.torch_utils import unpad_vtensor
from common.xdict import xdict
from src.utils.loss_modules import contact_deviation
from src.utils.mdev import eval_motion_deviation


def compute_avg_err(gt_dist, pred_dist, is_valid):
    assert len(gt_dist) == len(pred_dist)
    diff_list = []
    for gt, pred, valid in zip(gt_dist, pred_dist, is_valid):
        if valid:
            diff = torch.abs(gt - pred).mean()
        else:
            diff = torch.tensor(float("nan"))
        diff_list.append(diff)
    diff_list = torch.stack(diff_list).view(-1)
    assert len(diff_list) == len(gt_dist)
    return diff_list


def eval_field_errors(_pred, _targets, _meta_info):
    pred = copy.deepcopy(_pred).to("cpu")
    targets = copy.deepcopy(_targets).to("cpu")
    meta_info = copy.deepcopy(_meta_info).to("cpu")

    targets.overwrite(
        "dist.or", unpad_vtensor(targets["dist.or"], meta_info["object.v_len"])
    )
    targets.overwrite(
        "dist.ol", unpad_vtensor(targets["dist.ol"], meta_info["object.v_len"])
    )
    pred.overwrite("dist.or", unpad_vtensor(pred["dist.or"], meta_info["object.v_len"]))
    pred.overwrite("dist.ol", unpad_vtensor(pred["dist.ol"], meta_info["object.v_len"]))

    keys = ["dist.ro", "dist.lo", "dist.or", "dist.ol"]
    is_valid = _targets["is_valid"].bool().tolist()

    # validty of hand is not in use as if hand is out of frame  model should predict longer distance
    metric_dict = xdict(
        {
            key.replace("dist.", "avg/"): compute_avg_err(
                targets[key], pred[key], is_valid
            )
            for key in keys
        }
    )

    avg_ho_all = torch.stack((metric_dict["avg/ro"], metric_dict["avg/lo"]), dim=1)
    avg_oh_all = torch.stack((metric_dict["avg/or"], metric_dict["avg/ol"]), dim=1)

    avg_ho_all = torch_utils.nanmean(avg_ho_all, dim=1)
    avg_oh_all = torch_utils.nanmean(avg_oh_all, dim=1)

    metric_dict["avg/ho"] = avg_ho_all
    metric_dict["avg/oh"] = avg_oh_all
    metric_dict.pop("avg/ro", None)
    metric_dict.pop("avg/lo", None)
    metric_dict.pop("avg/or", None)
    metric_dict.pop("avg/ol", None)
    metric_dict = metric_dict.mul(1000.0).to_np()
    return metric_dict


def eval_degree(pred, targets, meta_info):
    is_valid = targets["is_valid"]

    # only evaluate on sequences with articulation
    invalid_idx = (1.0 - is_valid).long().nonzero().view(-1).cpu()

    pred_radian = pred["object.radian"].view(-1)  # radian
    gt_radian = targets["object.radian"].view(-1)  # radian
    arti_err = metrics.compute_arti_deg_error(pred_radian, gt_radian)

    # flag down sequences without articulation
    arti_err[invalid_idx] = float("nan")

    metric_dict = {}
    metric_dict["aae"] = arti_err
    return metric_dict


def eval_mpjpe_ra(pred, targets, meta_info):
    joints3d_cam_r_gt = targets["mano.j3d.cam.r"]
    joints3d_cam_l_gt = targets["mano.j3d.cam.l"]
    joints3d_cam_r_pred = pred["mano.j3d.cam.r"]
    joints3d_cam_l_pred = pred["mano.j3d.cam.l"]
    is_valid = targets["is_valid"]
    left_valid = targets["left_valid"] * is_valid
    right_valid = targets["right_valid"] * is_valid
    num_examples = len(joints3d_cam_r_gt)

    joints3d_cam_r_gt_ra = joints3d_cam_r_gt - joints3d_cam_r_gt[:, :1, :]
    joints3d_cam_l_gt_ra = joints3d_cam_l_gt - joints3d_cam_l_gt[:, :1, :]
    joints3d_cam_r_pred_ra = joints3d_cam_r_pred - joints3d_cam_r_pred[:, :1, :]
    joints3d_cam_l_pred_ra = joints3d_cam_l_pred - joints3d_cam_l_pred[:, :1, :]
    mpjpe_ra_r = metrics.compute_joint3d_error(
        joints3d_cam_r_gt_ra, joints3d_cam_r_pred_ra, right_valid
    )
    mpjpe_ra_l = metrics.compute_joint3d_error(
        joints3d_cam_l_gt_ra, joints3d_cam_l_pred_ra, left_valid
    )

    mpjpe_ra_r = mpjpe_ra_r.mean(axis=1)
    mpjpe_ra_l = mpjpe_ra_l.mean(axis=1)

    # average over hand direction
    mpjpe_ra_h = torch.FloatTensor(np.stack((mpjpe_ra_r, mpjpe_ra_l), axis=1))
    mpjpe_ra_h = torch_utils.nanmean(mpjpe_ra_h, dim=1)

    metric_dict = xdict()
    # metric_dict["mpjpe/ra/r"] = mpjpe_ra_r
    # metric_dict["mpjpe/ra/l"] = mpjpe_ra_l
    metric_dict["mpjpe/ra/h"] = mpjpe_ra_h
    metric_dict = metric_dict.mul(1000.0).to_np()

    # assert len(metric_dict["mpjpe/ra/r"]) == num_examples
    # assert len(metric_dict["mpjpe/ra/l"]) == num_examples
    assert len(metric_dict["mpjpe/ra/h"]) == num_examples
    return metric_dict

def compute_similarity_transform(S1, S2): # taken from https://github.com/miraymen/3dpw-eval/blob/master/evaluate.py
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    try: # SVD doesn't converge sometimes
        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale*(R.dot(mu1))

        # 7. Error:
        S1_hat = scale*R.dot(S1) + t
    except:
        S1_hat = np.ones_like(S1) * np.nan

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_errors(gt3ds, preds, valid):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3 root-aligned GT 3D joints
      - preds: N x 14 x 3 root-aligned predicted 3D joints
      - valid: whether the hand is visible
    """
    # convert to numpy
    gt3ds = gt3ds.cpu().numpy()
    preds = preds.cpu().numpy()
    valid = valid.cpu().numpy()
    errors, errors_pa = [], []
    for i, (gt3d, pred3d) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        pred3d = pred3d.reshape(-1, 3)

        # # Root align.
        # gt3d = align_by_pelvis(gt3d)
        # pred3d = align_by_pelvis(pred3d)

        # joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        # errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error)*valid[i]) # should be np.mean
    errors_pa = np.array(errors_pa)
    return errors_pa

def eval_mpjpe_pa_ra(pred, targets, meta_info): # refer to https://github.com/miraymen/3dpw-eval/blob/master/evaluate.py
    joints3d_cam_r_gt = targets["mano.j3d.cam.r"]
    joints3d_cam_l_gt = targets["mano.j3d.cam.l"]
    joints3d_cam_r_pred = pred["mano.j3d.cam.r"]
    joints3d_cam_l_pred = pred["mano.j3d.cam.l"]
    is_valid = targets["is_valid"]
    left_valid = targets["left_valid"] * is_valid
    right_valid = targets["right_valid"] * is_valid
    num_examples = len(joints3d_cam_r_gt)

    if 'egoexo' in meta_info['dataset']:
        left_valid_3d = targets['joints3d_valid_l'].bool()
        right_valid_3d = targets['joints3d_valid_r'].bool()
        bz = joints3d_cam_r_gt.shape[0]
        mpjpe_pa_ra_r, mpjpe_pa_ra_l = [], []
        mpjpe_ra_r, mpjpe_ra_l = [], []
        mpjpe_abs_r, mpjpe_abs_l = [], []
        for idx in range(bz):
            curr_r_gt = joints3d_cam_r_gt[idx][right_valid_3d[idx]]
            curr_r_pred = joints3d_cam_r_pred[idx][right_valid_3d[idx]]

            if len(curr_r_gt) == 0:
                mpjpe_ra_r.append(np.nan)
                mpjpe_pa_ra_r.append(np.nan)
                mpjpe_abs_r.append(np.nan)
                continue

            curr_abs_r_error = np.sqrt(np.sum((curr_r_gt.cpu().numpy().copy() - curr_r_pred.cpu().numpy().copy())**2, axis=1))
            mpjpe_abs_r.append(np.mean(curr_abs_r_error))

            # align by root joint
            curr_r_gt = curr_r_gt - curr_r_gt[0]
            curr_r_pred = curr_r_pred - curr_r_pred[0]

            curr_r_error = np.sqrt(np.sum((curr_r_gt.cpu().numpy() - curr_r_pred.cpu().numpy())**2, axis=1))
            mpjpe_ra_r.append(np.mean(curr_r_error))

            # compute errors
            curr_mpjpe_ra_r = compute_errors(curr_r_gt.unsqueeze(0), curr_r_pred.unsqueeze(0), right_valid[idx].unsqueeze(0))
            mpjpe_pa_ra_r.append(np.mean(curr_mpjpe_ra_r))

        for idx in range(bz):
            curr_l_gt = joints3d_cam_l_gt[idx][left_valid_3d[idx]]
            curr_l_pred = joints3d_cam_l_pred[idx][left_valid_3d[idx]]

            if len(curr_l_gt) == 0:
                mpjpe_ra_l.append(np.nan)
                mpjpe_pa_ra_l.append(np.nan)
                mpjpe_abs_l.append(np.nan)
                continue

            curr_abs_l_error = np.sqrt(np.sum((curr_l_gt.cpu().numpy().copy() - curr_l_pred.cpu().numpy().copy())**2, axis=1))
            mpjpe_abs_l.append(np.mean(curr_abs_l_error))
            
            curr_l_gt = curr_l_gt - curr_l_gt[0]
            curr_l_pred = curr_l_pred - curr_l_pred[0]

            curr_l_error = np.sqrt(np.sum((curr_l_gt.cpu().numpy() - curr_l_pred.cpu().numpy())**2, axis=1))
            mpjpe_ra_l.append(np.mean(curr_l_error))

            curr_mpjpe_ra_l = compute_errors(curr_l_gt.unsqueeze(0), curr_l_pred.unsqueeze(0), left_valid[idx].unsqueeze(0))
            mpjpe_pa_ra_l.append(np.mean(curr_mpjpe_ra_l))
        
        mpjpe_ra_r = np.array(mpjpe_ra_r)
        mpjpe_ra_l = np.array(mpjpe_ra_l)
        mpjpe_pa_ra_r = np.array(mpjpe_pa_ra_r)
        mpjpe_pa_ra_l = np.array(mpjpe_pa_ra_l)
        
        mpjpe_abs_r = np.array(mpjpe_abs_r)
        mpjpe_abs_l = np.array(mpjpe_abs_l)

        # average over hand direction
        mpjpe_ra_h = torch.FloatTensor(np.stack((mpjpe_ra_r, mpjpe_ra_l), axis=1))
        mpjpe_ra_h = torch_utils.nanmean(mpjpe_ra_h, dim=1)
        mpjpe_pa_ra_h = torch.FloatTensor(np.stack((mpjpe_pa_ra_r, mpjpe_pa_ra_l), axis=1))
        mpjpe_pa_ra_h = torch_utils.nanmean(mpjpe_pa_ra_h, dim=1)
        mpjpe_abs_h = torch.FloatTensor(np.stack((mpjpe_abs_r, mpjpe_abs_l), axis=1))
        mpjpe_abs_h = torch_utils.nanmean(mpjpe_abs_h, dim=1)

        metric_dict = xdict()
        metric_dict["mpjpe/pa/rao/r"] = mpjpe_ra_r
        metric_dict["mpjpe/pa/rao/l"] = mpjpe_ra_l
        metric_dict["mpjpe/pa/rao/h"] = mpjpe_ra_h

        metric_dict["mpjpe/pa/abs/r"] = mpjpe_abs_r
        metric_dict["mpjpe/pa/abs/l"] = mpjpe_abs_l
        metric_dict["mpjpe/pa/abs/h"] = mpjpe_abs_h

        metric_dict["mpjpe/pa/ra/r"] = mpjpe_pa_ra_r
        metric_dict["mpjpe/pa/ra/l"] = mpjpe_pa_ra_l
        metric_dict["mpjpe/pa/ra/h"] = mpjpe_pa_ra_h
        metric_dict = metric_dict.mul(1000.0).to_np()

        assert len(metric_dict["mpjpe/pa/ra/r"]) == num_examples
        assert len(metric_dict["mpjpe/pa/ra/l"]) == num_examples
        assert len(metric_dict["mpjpe/pa/ra/h"]) == num_examples
        return metric_dict
    
    else:
        # this only works when all 21 joints are available
        joints3d_cam_r_gt_ra = joints3d_cam_r_gt - joints3d_cam_r_gt[:, :1, :]
        joints3d_cam_l_gt_ra = joints3d_cam_l_gt - joints3d_cam_l_gt[:, :1, :]
        joints3d_cam_r_pred_ra = joints3d_cam_r_pred - joints3d_cam_r_pred[:, :1, :]
        joints3d_cam_l_pred_ra = joints3d_cam_l_pred - joints3d_cam_l_pred[:, :1, :]
        
        mpjpe_ra_r = compute_errors(joints3d_cam_r_gt_ra, joints3d_cam_r_pred_ra, right_valid)
        mpjpe_ra_l = compute_errors(joints3d_cam_l_gt_ra, joints3d_cam_l_pred_ra, left_valid)

        # average over hand direction
        mpjpe_ra_h = torch.FloatTensor(np.stack((mpjpe_ra_r, mpjpe_ra_l), axis=1))
        mpjpe_ra_h = torch_utils.nanmean(mpjpe_ra_h, dim=1)

        metric_dict = xdict()
        metric_dict["mpjpe/pa/ra/r"] = mpjpe_ra_r
        metric_dict["mpjpe/pa/ra/l"] = mpjpe_ra_l
        metric_dict["mpjpe/pa/ra/h"] = mpjpe_ra_h
        metric_dict = metric_dict.mul(1000.0).to_np()

        assert len(metric_dict["mpjpe/pa/ra/r"]) == num_examples
        assert len(metric_dict["mpjpe/pa/ra/l"]) == num_examples
        assert len(metric_dict["mpjpe/pa/ra/h"]) == num_examples
        return metric_dict


def eval_mrrpe(pred, targets, meta_info):
    joints3d_cam_r_gt = targets["mano.j3d.cam.r"]
    joints3d_cam_l_gt = targets["mano.j3d.cam.l"]
    joints3d_cam_r_pred = pred["mano.j3d.cam.r"]
    joints3d_cam_l_pred = pred["mano.j3d.cam.l"]
    v3d_cam_gt = unpad_vtensor(targets["object.v.cam"], targets["object.v_len"])
    v3d_cam_pred = unpad_vtensor(pred["object.v.cam"], targets["object.v_len"])

    bottom_idx = meta_info["part_ids"] == 2
    bottom_idx = [bidx.nonzero().view(-1) for bidx in bottom_idx]
    v3d_root_gt = [
        v3d_gt[bidx].mean(dim=0) for v3d_gt, bidx in zip(v3d_cam_gt, bottom_idx)
    ]
    v3d_root_pred = [
        v3d_pred[bidx].mean(dim=0) for v3d_pred, bidx in zip(v3d_cam_pred, bottom_idx)
    ]

    is_valid = targets["is_valid"]
    left_valid = targets["left_valid"] * is_valid
    right_valid = targets["right_valid"] * is_valid

    root_r_gt = joints3d_cam_r_gt[:, 0]
    root_l_gt = joints3d_cam_l_gt[:, 0]
    root_r_pred = joints3d_cam_r_pred[:, 0]
    root_l_pred = joints3d_cam_l_pred[:, 0]
    v3d_root_gt = torch.stack(v3d_root_gt, dim=0)
    v3d_root_pred = torch.stack(v3d_root_pred, dim=0)

    mrrpe_rl = metrics.compute_mrrpe(
        root_r_gt, root_l_gt, root_r_pred, root_l_pred, left_valid * right_valid
    )
    mrrpe_ro = metrics.compute_mrrpe(
        root_r_gt, v3d_root_gt, root_r_pred, v3d_root_pred, right_valid * is_valid
    )
    metric_dict = xdict()
    metric_dict["mrrpe/r/l"] = mrrpe_rl
    metric_dict["mrrpe/r/o"] = mrrpe_ro
    metric_dict = metric_dict.mul(1000.0).to_np()
    return metric_dict


def eval_mrrpe_hand(pred, targets, meta_info):
    joints3d_cam_r_gt = targets["mano.j3d.cam.r"]
    joints3d_cam_l_gt = targets["mano.j3d.cam.l"]
    joints3d_cam_r_pred = pred["mano.j3d.cam.r"]
    joints3d_cam_l_pred = pred["mano.j3d.cam.l"]

    is_valid = targets["is_valid"]
    left_valid = targets["left_valid"] * is_valid
    right_valid = targets["right_valid"] * is_valid

    root_r_gt = joints3d_cam_r_gt[:, 0]
    root_l_gt = joints3d_cam_l_gt[:, 0]
    root_r_pred = joints3d_cam_r_pred[:, 0]
    root_l_pred = joints3d_cam_l_pred[:, 0]

    mrrpe_rl = metrics.compute_mrrpe(
        root_r_gt, root_l_gt, root_r_pred, root_l_pred, left_valid * right_valid
    )
    metric_dict = xdict()
    metric_dict["mrrpe/r/l"] = mrrpe_rl
    metric_dict = metric_dict.mul(1000.0).to_np()
    return metric_dict


def eval_pixel_error(preds, targets, meta_info):
    j2d_pix_r_gt = targets['mano.j2d.r']
    j2d_pix_l_gt = targets['mano.j2d.l']
    j2d_pix_r_pred = preds['mano.j2d.r']
    j2d_pix_l_pred = preds['mano.j2d.l']

    is_valid = targets["is_valid"]
    left_valid = targets['joints_valid_l'] * (targets["left_valid"] * is_valid).view(-1,1)
    right_valid = targets['joints_valid_r'] * (targets["right_valid"] * is_valid).view(-1,1)

    pix_r = metrics.compute_pixel_error(j2d_pix_r_gt, j2d_pix_r_pred, right_valid)
    pix_l = metrics.compute_pixel_error(j2d_pix_l_gt, j2d_pix_l_pred, left_valid)

    metric_dict = xdict()
    metric_dict["pix_err/r"] = pix_r
    metric_dict["pix_err/l"] = pix_l
    metric_dict['pix_err/h'] = np.concatenate((pix_r, pix_l), axis=1)
    metric_dict = metric_dict.to_np()
    return metric_dict


def eval_v2v_success(pred, targets, meta_info):
    is_valid = targets["is_valid"]

    v3d_cam_gt = unpad_vtensor(targets["object.v.cam"], targets["object.v_len"])
    v3d_cam_pred = unpad_vtensor(pred["object.v.cam"], targets["object.v_len"])

    bottom_idx = meta_info["part_ids"] == 2
    bottom_idx = [bidx.nonzero().view(-1) for bidx in bottom_idx]
    v3d_root_gt = [
        v3d_gt[bidx].mean(dim=0) for v3d_gt, bidx in zip(v3d_cam_gt, bottom_idx)
    ]
    v3d_root_pred = [
        v3d_pred[bidx].mean(dim=0) for v3d_pred, bidx in zip(v3d_cam_pred, bottom_idx)
    ]

    v3d_cam_gt_ra = [
        v3d_gt - root[None, :] for v3d_gt, root in zip(v3d_cam_gt, v3d_root_gt)
    ]

    v3d_cam_pred_ra = [
        v3d_pred - root[None, :] for v3d_pred, root in zip(v3d_cam_pred, v3d_root_pred)
    ]

    v2v_ra = metrics.compute_v2v_dist_no_reduce(
        v3d_cam_gt_ra, v3d_cam_pred_ra, is_valid
    )

    diameters = meta_info["diameter"].cpu().numpy()

    alphas = [0.03, 0.05, 0.1]
    alphas = [0.05]
    metric_dict = xdict()
    for alpha in alphas:
        v2v_rate_ra_list = []
        for _v2v_ra, _diameter, _is_valid in zip(v2v_ra, diameters, is_valid):
            if bool(_is_valid):
                v2v_rate_ra = (_v2v_ra < _diameter * alpha).astype(np.float32)
                success = v2v_rate_ra.sum()
                v2v_rate_ra = success / v2v_rate_ra.shape[0]
                v2v_rate_ra_list.append(v2v_rate_ra)
            else:
                v2v_rate_ra_list.append(float("nan"))
        # percentage
        metric_dict[f"success_rate/{alpha:.2f}"] = np.array(v2v_rate_ra_list)
    metric_dict = metric_dict.mul(100.0).to_np()
    return metric_dict


def eval_contact_deviation(pred, targets, meta_info):
    cd_ro = contact_deviation(
        pred["object.v.cam"],
        pred["mano.v3d.cam.r"],
        targets["dist.ro"],
        targets["idx.ro"],
        targets["is_valid"],
        targets["right_valid"],
    )

    cd_lo = contact_deviation(
        pred["object.v.cam"],
        pred["mano.v3d.cam.l"],
        targets["dist.lo"],
        targets["idx.lo"],
        targets["is_valid"],
        targets["left_valid"],
    )
    cd_ho = torch.stack((cd_ro, cd_lo), dim=1)
    cd_ho = torch_utils.nanmean(cd_ho, dim=1)

    metric_dict = xdict()
    # metric_dict["cdev/ro"] = cd_ro
    # metric_dict["cdev/lo"] = cd_lo
    metric_dict["cdev/ho"] = cd_ho
    metric_dict = metric_dict.mul(1000)  # mm
    return metric_dict


def compute_error_accel(joints_gt, joints_pred, fps=30.0):
    """
    Computes acceleration error:
        First apply a center difference filter [1, -2, 1] along the seq
        Then divided by the stencil with h^2 where h =1/fps (second)
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).

    Modified from: https://github.com/mkocabas/VIBE/blob/master/lib/utils/eval_utils.py#L22
    Note: VIBE does not divide by the stencil h^2, so their results are not in mm instead of m/s^2
    """

    h = 1 / fps  # stencil width

    # (N-2)x14x3
    # m/s^2
    accel_gt = (joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]) / (h**2)
    accel_pred = (joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]) / (h**2)
    normed = torch.norm(accel_pred - accel_gt, dim=2)
    acc_err = torch.mean(normed, dim=1)
    return acc_err


def eval_acc_pose(pred, targets, meta_info):
    gt_vo = targets["object.v.cam"]
    gt_vr = targets["mano.v3d.cam.r"]
    gt_vl = targets["mano.v3d.cam.l"]

    pred_vo = pred["object.v.cam"]
    pred_vr = pred["mano.v3d.cam.r"]
    pred_vl = pred["mano.v3d.cam.l"]

    num_frames = gt_vo.shape[0]

    # hand roots
    pred_root_r = pred["mano.j3d.cam.r"][:, :1]
    pred_root_l = pred["mano.j3d.cam.l"][:, :1]
    gt_root_r = targets["mano.j3d.cam.r"][:, :1]
    gt_root_l = targets["mano.j3d.cam.l"][:, :1]

    # object roots
    parts_ids = targets["object.parts_ids"]
    bottom_idx = parts_ids[0] == 2
    gt_root_o = gt_vo[:, bottom_idx].mean(dim=1)[:, None, :]
    pred_root_o = pred_vo[:, bottom_idx].mean(dim=1)[:, None, :]

    # root relative (num_frames, num_verts, 3)
    gt_vr_ra = gt_vr - gt_root_r
    gt_vl_ra = gt_vl - gt_root_l
    gt_vo_ra = gt_vo - gt_root_o

    # root relative (num_frames, num_verts, 3)
    pred_vr_ra = pred_vr - pred_root_r
    pred_vl_ra = pred_vl - pred_root_l
    pred_vo_ra = pred_vo - pred_root_o

    # m/s^2
    acc_r = compute_error_accel(gt_vr_ra, pred_vr_ra)
    acc_l = compute_error_accel(gt_vl_ra, pred_vl_ra)
    acc_o = compute_error_accel(gt_vo_ra, pred_vo_ra)

    is_valid = targets["is_valid"]
    left_valid = targets["left_valid"] * is_valid
    right_valid = targets["right_valid"] * is_valid

    is_valid = is_valid.cpu().numpy()
    left_valid = left_valid.cpu().numpy()
    right_valid = right_valid.cpu().numpy()

    # acc of time step t is valid if {t-1, t, t+1} are valid
    acc_valid_r = (
        np.convolve(right_valid, np.ones(3), mode="valid").astype(np.int64) == 3
    )
    acc_valid_l = (
        np.convolve(left_valid, np.ones(3), mode="valid").astype(np.int64) == 3
    )
    acc_valid_o = np.convolve(is_valid, np.ones(3), mode="valid").astype(np.int64) == 3

    # set invalid acc to nan
    acc_r[~acc_valid_r] = float("nan")
    acc_l[~acc_valid_l] = float("nan")
    acc_o[~acc_valid_o] = float("nan")

    # average by hands
    acc_h = torch.stack((acc_r, acc_l), dim=1)
    acc_h = torch_utils.nanmean(acc_h, dim=1)

    # pad nan to start and end of tensor
    acc_r = torch.cat(
        (torch.tensor([float("nan")]), acc_r, torch.tensor([float("nan")]))
    )
    acc_l = torch.cat(
        (torch.tensor([float("nan")]), acc_l, torch.tensor([float("nan")]))
    )
    acc_h = torch.cat(
        (torch.tensor([float("nan")]), acc_h, torch.tensor([float("nan")]))
    )

    metric_dict = xdict()
    # metric_dict["acc/r"] = acc_r
    # metric_dict["acc/l"] = acc_l
    metric_dict["acc/h"] = acc_h
    metric_dict["acc/o"] = acc_o
    metric_dict = metric_dict.to_np()  # m/s^2

    # assert metric_dict["acc/r"].shape[0] == num_frames
    # assert metric_dict["acc/l"].shape[0] == num_frames
    assert metric_dict["acc/h"].shape[0] == num_frames
    return metric_dict


def eval_acc_field(pred, targets, meta_info):
    is_valid = targets["is_valid"]
    right_valid = targets["right_valid"] * is_valid
    left_valid = targets["left_valid"] * is_valid
    num_frames = is_valid.shape[0]

    targets_dist_lo = targets["dist.lo"][:, :, None].clone()
    targets_dist_ro = targets["dist.ro"][:, :, None].clone()
    targets_dist_ol = targets["dist.ol"][:, :, None].clone()
    targets_dist_or = targets["dist.or"][:, :, None].clone()
    num_verts = targets_dist_ol.shape[1]
    assert targets_dist_or.shape[1] == num_verts

    pred_dist_lo = pred["dist.lo"][:, :, None].clone()
    pred_dist_ro = pred["dist.ro"][:, :, None].clone()
    pred_dist_ol = pred["dist.ol"][:, :num_verts, None].clone()
    pred_dist_or = pred["dist.or"][:, :num_verts, None].clone()

    acc_lo = compute_error_accel(targets_dist_lo, pred_dist_lo)
    acc_ro = compute_error_accel(targets_dist_ro, pred_dist_ro)
    acc_ol = compute_error_accel(targets_dist_ol, pred_dist_ol)
    acc_or = compute_error_accel(targets_dist_or, pred_dist_or)

    is_valid = is_valid.cpu().numpy()
    left_valid = left_valid.cpu().numpy()
    right_valid = right_valid.cpu().numpy()

    # acc is valid if {t-1, t, t+1} are valid for numerical differentiation
    acc_valid_r = (
        np.convolve(right_valid, np.ones(3), mode="valid").astype(np.int64) == 3
    )
    acc_valid_l = (
        np.convolve(left_valid, np.ones(3), mode="valid").astype(np.int64) == 3
    )
    acc_valid_o = np.convolve(is_valid, np.ones(3), mode="valid").astype(np.int64) == 3

    acc_ro[~acc_valid_r] = float("nan")
    acc_lo[~acc_valid_l] = float("nan")
    acc_or[~acc_valid_o] = float("nan")
    acc_ol[~acc_valid_o] = float("nan")

    acc_ho = torch.stack((acc_ro, acc_lo), dim=1)
    acc_oh = torch.stack((acc_or, acc_ol), dim=1)
    acc_ho = torch_utils.nanmean(acc_ho, dim=1)
    acc_oh = torch_utils.nanmean(acc_oh, dim=1)

    # pad nan
    acc_ro = torch.cat(
        (torch.tensor([float("nan")]), acc_ro, torch.tensor([float("nan")]))
    )
    acc_lo = torch.cat(
        (torch.tensor([float("nan")]), acc_lo, torch.tensor([float("nan")]))
    )
    acc_or = torch.cat(
        (torch.tensor([float("nan")]), acc_or, torch.tensor([float("nan")]))
    )
    acc_ol = torch.cat(
        (torch.tensor([float("nan")]), acc_ol, torch.tensor([float("nan")]))
    )
    acc_oh = torch.cat(
        (torch.tensor([float("nan")]), acc_oh, torch.tensor([float("nan")]))
    )
    acc_ho = torch.cat(
        (torch.tensor([float("nan")]), acc_ho, torch.tensor([float("nan")]))
    )

    metric_dict = xdict()
    # metric_dict["acc/ro"] = acc_ro
    # metric_dict["acc/lo"] = acc_lo
    # metric_dict["acc/or"] = acc_or
    # metric_dict["acc/ol"] = acc_ol
    metric_dict["acc/oh"] = acc_oh
    metric_dict["acc/ho"] = acc_ho

    # assert metric_dict["acc/ro"].shape[0] == num_frames
    # assert metric_dict["acc/lo"].shape[0] == num_frames
    # assert metric_dict["acc/or"].shape[0] == num_frames
    # assert metric_dict["acc/ol"].shape[0] == num_frames
    assert metric_dict["acc/oh"].shape[0] == num_frames
    assert metric_dict["acc/ho"].shape[0] == num_frames
    return metric_dict


eval_fn_dict = {
    "aae": eval_degree,
    "mpjpe.ra": eval_mpjpe_ra,
    "mpjpe.mano": eval_mpjpe_mano,
    "mpjpe.pa.ra": eval_mpjpe_pa_ra,
    "mrrpe": eval_mrrpe,
    "mrrpe.rl": eval_mrrpe_hand,
    "success_rate": eval_v2v_success,
    "avg_err_field": eval_field_errors,
    "cdev": eval_contact_deviation,
    "mdev": eval_motion_deviation,
    "acc_err_pose": eval_acc_pose,
    "acc_err_field": eval_acc_field,
    "pix_err": eval_pixel_error,
}
