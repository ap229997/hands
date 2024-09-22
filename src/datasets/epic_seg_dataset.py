import os
import pickle
import cv2
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import common.data_utils as data_utils
import common.rot as rot
import common.transforms as tf
import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
from common.object_tensors import ObjectTensors
from src.datasets.dataset_utils import get_valid, pad_jts2d, downsample


def dummy_joint_data():
    bbox = None
    joints = list(np.zeros((21,2)))
    joints_valid = [0]*21
    return {'bbox': bbox, 'joints': joints, 'joints_valid': joints_valid}

class EPICSegDataset(Dataset):
    def __init__(self, args, mode='train') -> None:
        super().__init__()

        filename = f"{os.environ['DATA_DIR']}/epic_hands/modal_amodal_annot.pkl"
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        self.args = args
        self.aug_data = 'train' in mode
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)
        self.egocam_k = None

        bbox_file = f"{os.environ['DATA_DIR']}/epic_hands/grasp_visor_train.pkl"
        with open(bbox_file, 'rb') as f:
            self.bbox_data = pickle.load(f)

        if args.get('use_gt_hand_mask', False):
            masks_npz_file = f"{os.environ['DATA_DIR']}/epic_hands/visor_masks_train.npz"
        else:
            masks_npz_file = f"{os.environ['DATA_DIR']}/epic_hands/visor_pred_masks_train.npz"
        self.masks_npz = np.load(masks_npz_file, allow_pickle=True)

        self.imgnames = list(set(self.data.keys()) & set(self.bbox_data.keys()) & set(self.masks_npz.files))

        # subsample indices
        all_keys = list(range(len(self.imgnames)))
        self.subsampled_keys = sorted(downsample(all_keys, mode))

        logger.info("# samples in EPIC Seg %s: %d" % (mode, len(self.subsampled_keys)))

    def __len__(self):
        if self.args.debug:
            return 11
        return len(self.subsampled_keys)

    def __getitem__(self, index):
        idx = self.subsampled_keys[index]
        imgname = self.imgnames[idx]
        args = self.args
        
        if args.get('use_render_seg_loss', False):
            # some issue loading P22 seg masks from VISOR
            is_not_corrupt = False
            while not is_not_corrupt:
                try:
                    mask_npz = self.masks_npz[imgname][...,0] # only R channel is colored
                    is_not_corrupt = True
                except:
                    idx = random.choice(self.subsampled_keys)
                    imgname = self.imgnames[idx]

        data = self.bbox_data[imgname]
        modal_labels = self.data[imgname] # 0: occluded, 1: unoccluded
        if 'left' not in modal_labels:
            modal_labels['left'] = 0
        if 'right' not in modal_labels:
            modal_labels['right'] = 0

        image_size = {"width": 1920, "height": 1080}

        bbox = [image_size['width'] / 2, image_size['height'] / 2, max(image_size['width'], image_size['height']) / 200] # original bbox
        is_egocam = True

        cv_img, img_status = read_img(imgname, (2800, 2000, 3)) # dummy value for shape sufficies

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        # no scaling for egocam to make intrinsics consistent
        use_gt_k = True
        augm_dict["sc"] = 1.0

        img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )
        
        # in string format, change later
        right_bbox = data['right_bbox'] # (x0, y0, x1, y1)
        left_bbox = data['left_bbox'] # (x0, y0, x1, y1)
        
        if right_bbox is None:
            r_bbox_og = np.array([0, 0, args.img_res-1, args.img_res-1])
        else:
            right_bbox = np.array(right_bbox).astype(np.float32)
            r_bbox_og = right_bbox.copy().astype(np.int16)
        if left_bbox is None:
            l_bbox_og = np.array([0, 0, args.img_res-1, args.img_res-1])
        else:
            left_bbox = np.array(left_bbox).astype(np.float32)
            l_bbox_og = left_bbox.copy().astype(np.int16)

        if args.get('use_render_seg_loss', False):
            mask_r = mask_npz==255
            mask_l = mask_npz==127
            mask_r = np.stack([mask_r]*3, axis=-1) # replicate to 3 channels
            mask_l = np.stack([mask_l]*3, axis=-1) # replicate to 3 channels
            mask_bbox_r = np.zeros_like(mask_r)
            mask_bbox_r[r_bbox_og[1]:r_bbox_og[3], r_bbox_og[0]:r_bbox_og[2], :] = 1 # be careful about height, width ordering
            mask_bbox_l = np.zeros_like(mask_l)
            mask_bbox_l[l_bbox_og[1]:l_bbox_og[3], l_bbox_og[0]:l_bbox_og[2], :] = 1 # be careful about height, width ordering
            mask_r = mask_r * mask_bbox_r
            mask_l = mask_l * mask_bbox_l
            mask_r = 255 * mask_r.astype(np.float32)
            mask_l = 255 * mask_l.astype(np.float32)

            # augment masks using the same augmentation parameters as the image
            augm_dict_mask = augm_dict.copy()
            augm_dict_mask["pn"] = np.ones_like(augm_dict_mask["pn"]) # its a mask, so no need to add noise to color
            mask_img_r = data_utils.mask_processing(
                    self.aug_data,
                    mask_r,
                    center,
                    scale,
                    augm_dict_mask,
                    img_res=args.img_res,
                )
            mask_img_l = data_utils.mask_processing(
                    self.aug_data,
                    mask_l,
                    center,
                    scale,
                    augm_dict_mask,
                    img_res=args.img_res,
                )

        if right_bbox is not None:
            end_pts = np.array([[right_bbox[0], right_bbox[1]], [right_bbox[2], right_bbox[3]]])
            end_pts = data_utils.j2d_processing(pad_jts2d(end_pts), center, scale, augm_dict, args.img_res)
            end_pts = ((end_pts[...,:2]+1)/2)*args.img_res
            end_pts = end_pts.flatten()
            right_bbox = [end_pts[0], end_pts[1], end_pts[2]-end_pts[0], end_pts[3]-end_pts[1]]
        if left_bbox is not None:
            end_pts = np.array([[left_bbox[0], left_bbox[1]], [left_bbox[2], left_bbox[3]]])
            end_pts = data_utils.j2d_processing(pad_jts2d(end_pts), center, scale, augm_dict, args.img_res)
            end_pts = ((end_pts[...,:2]+1)/2)*args.img_res
            end_pts = end_pts.flatten()
            left_bbox = [end_pts[0], end_pts[1], end_pts[2]-end_pts[0], end_pts[3]-end_pts[1]]

        bbox_scale = 1.5
        if args.get('bbox_scale', None) is not None:
            bbox_scale = args.bbox_scale
        r_img, r_bbox = data_utils.crop_and_pad(img, right_bbox, args, scale=bbox_scale)
        l_img, l_bbox = data_utils.crop_and_pad(img, left_bbox, args, scale=bbox_scale)
        norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
        norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

        img_ds = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [args.img_res/2, args.img_res/2, args.img_res, args.img_res], 
                            1.0, 0.0, [args.img_res_ds, args.img_res_ds], cv2.INTER_CUBIC)[0].transpose(2,0,1)
        img_ds = np.clip(img_ds, 0, 1)
        img_ds = torch.from_numpy(img_ds).float()
        norm_img = self.normalize_img(img_ds)

        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        inputs["r_img"] = norm_r_img
        inputs["l_img"] = norm_l_img
        inputs["r_bbox"] = r_bbox
        inputs["l_bbox"] = l_bbox
        if args.use_gt_bbox:
            inputs['r_bbox_og'] = r_bbox_og
            inputs['l_bbox_og'] = l_bbox_og

        # dummy values below
        targets["mano.j2d.norm.r"] = torch.zeros((21,2)).float()
        targets["mano.j2d.norm.l"] = torch.zeros((21,2)).float()
        targets["mano.pose.r"] = torch.zeros((48,)).float() # dummy values
        targets["mano.pose.l"] = torch.zeros((48,)).float() # dummy values
        targets["mano.beta.r"] = torch.tensor([0.82747316,  0.13775729, -0.39435294, 0.17889787, -0.73901576, 0.7788163, -0.5702684, 0.4947751, -0.24890041, 1.5943261]) # mean value from val set
        targets["mano.beta.l"] = torch.tensor([-0.19330633, -0.08867972, -2.5790455, -0.10344583, -0.71684015, -0.28285977, 0.55171007, -0.8403888, -0.8490544, -1.3397144]) # mean value from val set
        targets["mano.j3d.full.r"] = torch.zeros((21, 3)).float()
        targets["mano.j3d.full.l"] = torch.zeros((21, 3)).float()

        meta_info["imgname"] = imgname

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = np.zeros((3, 3)) # dummy intrinsic value, default focal length = 1000
        fixed_focal_length = args.focal_length * (args.img_res / max(image_size["width"], image_size["height"]))
        intrx = data_utils.get_aug_intrix(
            intrx,
            fixed_focal_length,
            args.img_res,
            False,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k
        else:
            intrx = intrx.numpy()
        if not isinstance(intrx, np.ndarray):
            intrx = intrx.numpy()

        intrx_for_enc = intrx.copy()
        if args.get('no_intrx', False):
            intrx_for_enc = np.eye(3)
            intrx_for_enc[0,0] = args.img_res / 2
            intrx_for_enc[1,1] = args.img_res / 2
            intrx_for_enc[0,2] = args.img_res / 2
            intrx_for_enc[1,2] = args.img_res / 2

        # TODO: abstract this to a function which can be used across different encodings and datasets
        if args.pos_enc is not None:
            L = args.n_freq_pos_enc
            
            if 'center' in args.pos_enc or 'perspective_correction' in args.pos_enc:
                # center of left & right image
                r_center = (r_bbox[:2] + r_bbox[2:]) / 2.0
                l_center = (l_bbox[:2] + l_bbox[2:]) / 2.0
                r_angle_x, r_angle_y = np.arctan2(r_center[0]-intrx_for_enc[0,2], intrx_for_enc[0,0]), np.arctan2(r_center[1]-intrx_for_enc[1,2], intrx_for_enc[1,1])
                l_angle_x, l_angle_y = np.arctan2(l_center[0]-intrx_for_enc[0,2], intrx_for_enc[0,0]), np.arctan2(l_center[1]-intrx_for_enc[1,2], intrx_for_enc[1,1])
                r_angle = np.array([r_angle_x, r_angle_y]).astype(np.float32)
                l_angle = np.array([l_angle_x, l_angle_y]).astype(np.float32)
                inputs["r_center_angle"], inputs["l_center_angle"] = r_angle, l_angle
                targets['center.r'], targets['center.l'] = r_angle, l_angle

            if 'corner' in args.pos_enc:
                # corners of left & right image
                r_corner = np.array([[r_bbox[0], r_bbox[1]], [r_bbox[0], r_bbox[3]], [r_bbox[2], r_bbox[1]], [r_bbox[2], r_bbox[3]]])
                l_corner = np.array([[l_bbox[0], l_bbox[1]], [l_bbox[0], l_bbox[3]], [l_bbox[2], l_bbox[1]], [l_bbox[2], l_bbox[3]]])
                r_corner = np.stack([r_corner[:,0]-intrx_for_enc[0,2], r_corner[:,1]-intrx_for_enc[1,2]], axis=-1)
                l_corner = np.stack([l_corner[:,0]-intrx_for_enc[0,2], l_corner[:,1]-intrx_for_enc[1,2]], axis=-1)
                r_angle = np.arctan2(r_corner, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).flatten().astype(np.float32)
                l_angle = np.arctan2(l_corner, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).flatten().astype(np.float32)
                inputs["r_corner_angle"], inputs["l_corner_angle"] = r_angle, l_angle
                targets['corner.r'], targets['corner.l'] = r_angle, l_angle

            if args.pos_enc is not None and 'dense' in args.pos_enc:
                # dense positional encoding for all pixels
                r_x_grid, r_y_grid = range(r_bbox[0], r_bbox[2]+1), range(r_bbox[1], r_bbox[3]+1)
                r_x_grid, r_y_grid = np.meshgrid(r_x_grid, r_y_grid, indexing='ij') # torch doesn't support batched meshgrid
                l_x_grid, l_y_grid = range(l_bbox[0], l_bbox[2]+1), range(l_bbox[1], l_bbox[3]+1)
                l_x_grid, l_y_grid = np.meshgrid(l_x_grid, l_y_grid, indexing='ij')
                r_pix = np.stack([r_x_grid-intrx_for_enc[0,2], r_y_grid-intrx_for_enc[1,2]], axis=-1)
                l_pix = np.stack([l_x_grid-intrx_for_enc[0,2], l_y_grid-intrx_for_enc[1,2]], axis=-1)
                r_angle = np.arctan2(r_pix, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).transpose(2,0,1).astype(np.float32)
                l_angle = np.arctan2(l_pix, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).transpose(2,0,1).astype(np.float32)
                
                r_angle_fdim = np.zeros((r_angle.shape[0], args.img_res, args.img_res))
                l_angle_fdim = np.zeros((l_angle.shape[0], args.img_res, args.img_res))
                r_angle_fdim[:r_angle.shape[0], :r_angle.shape[1], :r_angle.shape[2]] = r_angle
                l_angle_fdim[:l_angle.shape[0], :l_angle.shape[1], :l_angle.shape[2]] = l_angle
                inputs["r_dense_angle"], inputs["l_dense_angle"] = r_angle_fdim.astype(np.float32), l_angle_fdim.astype(np.float32)
                r_mask = np.zeros((args.img_res, args.img_res))
                r_mask[:r_angle.shape[1], :r_angle.shape[2]] = 1
                l_mask = np.zeros((args.img_res, args.img_res))
                l_mask[:l_angle.shape[1], :l_angle.shape[2]] = 1
                inputs["r_dense_mask"], inputs["l_dense_mask"] = r_mask.astype(np.float32), l_mask.astype(np.float32)

            if 'cam_conv' in args.pos_enc:
                # dense positional encoding for all pixels
                r_x_grid, r_y_grid = range(r_bbox[0], r_bbox[2]+1), range(r_bbox[1], r_bbox[3]+1)
                r_x_grid, r_y_grid = np.meshgrid(r_x_grid, r_y_grid, indexing='ij') # torch doesn't support batched meshgrid
                l_x_grid, l_y_grid = range(l_bbox[0], l_bbox[2]+1), range(l_bbox[1], l_bbox[3]+1)
                l_x_grid, l_y_grid = np.meshgrid(l_x_grid, l_y_grid, indexing='ij')
                r_pix = np.stack([r_x_grid-intrx_for_enc[0,2], r_y_grid-intrx_for_enc[1,2]], axis=-1)
                l_pix = np.stack([l_x_grid-intrx_for_enc[0,2], l_y_grid-intrx_for_enc[1,2]], axis=-1)
                r_pix_transp = r_pix.transpose(2,0,1).astype(np.float32)
                l_pix_transp = l_pix.transpose(2,0,1).astype(np.float32)
                r_pix_centered = np.stack([2*r_x_grid/args.img_res-1, 2*r_y_grid/args.img_res-1], axis=-1).transpose(2,0,1).astype(np.float32)
                l_pix_centered = np.stack([2*l_x_grid/args.img_res-1, 2*l_y_grid/args.img_res-1], axis=-1).transpose(2,0,1).astype(np.float32)
                r_angle = np.arctan2(r_pix, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).transpose(2,0,1).astype(np.float32)
                l_angle = np.arctan2(l_pix, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).transpose(2,0,1).astype(np.float32)
                
                r_angle_fdim = np.zeros((r_angle.shape[0]+r_pix_transp.shape[0]+r_pix_centered.shape[0], args.img_res, args.img_res))
                r_angle_fdim[:r_angle.shape[0], :r_angle.shape[1], :r_angle.shape[2]] = r_angle
                r_angle_fdim[r_angle.shape[0]:r_angle.shape[0]+r_pix_transp.shape[0], :r_angle.shape[1], :r_angle.shape[2]] = r_pix_transp
                r_angle_fdim[r_angle.shape[0]+r_pix_transp.shape[0]:, :r_angle.shape[1], :r_angle.shape[2]] = r_pix_centered

                l_angle_fdim = np.zeros((l_angle.shape[0]+l_pix_transp.shape[0]+r_pix_centered.shape[0], args.img_res, args.img_res))
                l_angle_fdim[:l_angle.shape[0], :l_angle.shape[1], :l_angle.shape[2]] = l_angle
                l_angle_fdim[l_angle.shape[0]:l_angle.shape[0]+l_pix_transp.shape[0], :l_angle.shape[1], :l_angle.shape[2]] = l_pix_transp
                l_angle_fdim[l_angle.shape[0]+l_pix_transp.shape[0]:, :l_angle.shape[1], :l_angle.shape[2]] = l_pix_centered

                inputs["r_dense_angle"], inputs["l_dense_angle"] = r_angle_fdim.astype(np.float32), l_angle_fdim.astype(np.float32)

                r_mask = np.zeros((args.img_res, args.img_res))
                r_mask[:r_angle.shape[1], :r_angle.shape[2]] = 1
                l_mask = np.zeros((args.img_res, args.img_res))
                l_mask[:l_angle.shape[1], :l_angle.shape[2]] = 1
                inputs["r_dense_mask"], inputs["l_dense_mask"] = r_mask.astype(np.float32), l_mask.astype(np.float32)

            if 'sinusoidal_cc' in args.pos_enc:
                # center of left & right image
                r_center = (r_bbox[:2] + r_bbox[2:]) / 2.0
                l_center = (l_bbox[:2] + l_bbox[2:]) / 2.0
                r_angle = 2*r_center/args.img_res - 1
                l_angle = 2*l_center/args.img_res - 1
                inputs["r_center_angle"], inputs["l_center_angle"] = r_angle, l_angle
                targets['center.r'], targets['center.l'] = r_angle, l_angle

                # corners of left & right image
                r_corner = np.array([[r_bbox[0], r_bbox[1]], [r_bbox[0], r_bbox[3]], [r_bbox[2], r_bbox[1]], [r_bbox[2], r_bbox[3]]])
                l_corner = np.array([[l_bbox[0], l_bbox[1]], [l_bbox[0], l_bbox[3]], [l_bbox[2], l_bbox[1]], [l_bbox[2], l_bbox[3]]])
                r_angle = 2*r_corner/args.img_res - 1
                l_angle = 2*l_corner/args.img_res - 1
                r_angle = r_angle.flatten()
                l_angle = l_angle.flatten()
                inputs["r_corner_angle"], inputs["l_corner_angle"] = r_angle, l_angle
                targets['corner.r'], targets['corner.l'] = r_angle, l_angle

            if 'pcl' in args.pos_enc: # https://github.com/yu-frank/PerspectiveCropLayers/blob/main/src/pcl_demo.ipynb

                # define functions for PCL
                def virtualCameraRotationFromPosition(position):
                    x, y, z = position[0], position[1], position[2]
                    n1x = math.sqrt(1 + x ** 2)
                    d1x = 1 / n1x
                    d1xy = 1 / math.sqrt(1 + x ** 2 + y ** 2)
                    d1xy1x = 1 / math.sqrt((1 + x ** 2 + y ** 2) * (1 + x ** 2))
                    R_virt2orig = np.array([d1x, -x * y * d1xy1x, x * d1xy,
                                            0*x,      n1x * d1xy, y * d1xy,
                                        -x * d1x,     -y * d1xy1x, 1 * d1xy]).reshape(3,3)
                    return R_virt2orig

                def bK_virt(p_position, K_c, bbox_size_img, focal_at_image_plane=True, slant_compensation=True):
                    p_length = np.linalg.norm(p_position)
                    focal_length_factor = 1
                    if focal_at_image_plane:
                        focal_length_factor *= p_length
                    if slant_compensation:
                        sx = 1.0 / math.sqrt(p_position[0]**2+p_position[2]**2)  # this is cos(phi)
                        sy = math.sqrt(p_position[0]**2+1) / math.sqrt(p_position[0]**2+p_position[1]**2 + 1)  # this is cos(theta)
                        bbox_size_img = np.array(bbox_size_img) * np.array([sx,sy])

                    f_orig = np.diag(K_c)[:2]
                    f_compensated = focal_length_factor * f_orig / bbox_size_img # dividing by the target bbox_size_img will make the coordinates normalized to 0..1, as needed for the perspective grid sample function; an alternative would be to make the grid_sample operate on pixel coordinates
                    K_virt = np.zeros((3,3))
                    K_virt[2,2] = 1
                    # Note, in unit image coordinates ranging from 0..1
                    K_virt[0, 0] = f_compensated[0]
                    K_virt[1, 1] = f_compensated[1]
                    K_virt[:2, 2] = 0.5
                    return K_virt

                def perspective_grid(P_virt2orig, image_pixel_size, crop_pixel_size_wh, transform_to_pytorch=False):
                    # create a grid of linearly increasing indices (one for each pixel, going from 0..1)
                    xs = torch.linspace(0, 1, crop_pixel_size_wh[0])
                    ys = torch.linspace(0, 1, crop_pixel_size_wh[1])

                    rs, cs = torch.meshgrid([xs, ys])
                    # cs = ys.view(1, -1).repeat(xs.size(0), 1)
                    # rs = xs.view(-1, 1).repeat(1, ys.size(0))
                    zs = torch.ones(rs.shape)  # init homogeneous coordinate to 1
                    pv = torch.stack([rs, cs, zs])

                    # expand along batch dimension
                    grid = pv.expand([3, crop_pixel_size_wh[0], crop_pixel_size_wh[1]])

                    # linearize the 2D grid to a single dimension, to apply transformation
                    bpv_lin = grid.view([3, -1])

                    # do the projection
                    bpv_lin_orig = torch.matmul(P_virt2orig, bpv_lin)
                    eps = 0.00000001
                    bpv_lin_orig_p = bpv_lin_orig[:2, :] / (eps + bpv_lin_orig[2:3, :]) # projection, divide homogeneous coord

                    # go back from linear to two–dimensional outline of points
                    bpv_orig = bpv_lin_orig_p.view(2, crop_pixel_size_wh[0], crop_pixel_size_wh[1])

                    # the sampling function assumes the position information on the last dimension
                    bpv_orig = bpv_orig.permute([2, 1, 0])

                    # the transformed points will be in pixel coordinates ranging from 0 up to the image width/height (unmapped from the original intrinsics matrix)
                    # but the pytorch grid_sample function assumes it in -1,..,1; the direction is already correct (assuming negative y axis, which is also assumed by bytorch)
                    if transform_to_pytorch:
                        bpv_orig /= image_pixel_size # map to 0..1
                        bpv_orig *= 2 # to 0...2
                        bpv_orig -= 1 # to -1...1

                    return bpv_orig

                r_c = (inputs['r_bbox'][:2]+inputs['r_bbox'][2:])/2
                l_c = (inputs['l_bbox'][:2]+inputs['l_bbox'][2:])/2
                r_w, r_h = inputs['r_bbox'][2]-inputs['r_bbox'][0], inputs['r_bbox'][3]-inputs['r_bbox'][1]
                l_w, l_h = inputs['l_bbox'][2]-inputs['l_bbox'][0], inputs['l_bbox'][3]-inputs['l_bbox'][1]
                K_inv = np.linalg.inv(intrx)
                r_pos = np.matmul(K_inv, np.array([r_c[0], r_c[1], 1]))
                l_pos = np.matmul(K_inv, np.array([l_c[0], l_c[1], 1]))
                
                def pcl_layer(p_pos, K_c, w_c, h_c):
                    # get rotation from orig to new coordinate frame
                    R_virt2orig = virtualCameraRotationFromPosition(p_pos)
                    # determine target frame
                    K_virt = bK_virt(p_pos, K_c, [w_c, h_c])
                    K_virt_inv = np.linalg.inv(K_virt)
                    # projective transformation orig to virtual camera
                    P_virt2orig = np.matmul(K_c, np.matmul(R_virt2orig, K_virt_inv))

                    # convert to torch
                    P_virt2orig = torch.from_numpy(P_virt2orig).float()
                    K_virt = torch.from_numpy(K_virt).float()
                    R_virt2orig = torch.from_numpy(R_virt2orig).float()

                    grid_perspective = perspective_grid(P_virt2orig, args.img_res, [w_c, h_c], transform_to_pytorch=True)
                    return grid_perspective, R_virt2orig

                r_size = max(r_w, r_h)
                if r_size == 0: r_size = args.img_res
                r_grid_perspective, R_virt2orig_r = pcl_layer(r_pos, intrx.copy(), r_size, r_size)
                l_size = max(l_w, l_h)
                if l_size == 0: l_size = args.img_res
                l_grid_perspective, R_virt2orig_l = pcl_layer(l_pos, intrx.copy(), l_size, l_size)
                    
                n_r_img = F.grid_sample(inputs['img'].unsqueeze(0), r_grid_perspective.unsqueeze(0))[0]
                n_l_img = F.grid_sample(inputs['img'].unsqueeze(0), l_grid_perspective.unsqueeze(0))[0]

                # resize n_r_img, n_l_img to args.img_res
                n_r_img = F.interpolate(n_r_img.unsqueeze(0), size=(args.img_res, args.img_res), mode='bilinear', align_corners=True)[0]
                n_l_img = F.interpolate(n_l_img.unsqueeze(0), size=(args.img_res, args.img_res), mode='bilinear', align_corners=True)[0]

                inputs['r_img'] = n_r_img
                inputs['l_img'] = n_l_img
                inputs['r_rot'] = R_virt2orig_r
                inputs['l_rot'] = R_virt2orig_l

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        meta_info['dataset'] = 'epic_seg'
        meta_info["dist"] = torch.FloatTensor(torch.zeros(8)) # dummy value
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])

        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = 0
        meta_info['is_j3d_loss'] = 0
        meta_info['is_beta_loss'] = 0
        meta_info['is_pose_loss'] = 0
        meta_info['is_cam_loss'] = 0
        meta_info['is_grasp_loss'] = 0

        meta_info['is_grasp_loss'] = 0
        targets["grasp.l"] = 8 # no grasp
        targets["grasp.r"] = 8 # no grasp
        targets['grasp_valid_r'] = 0
        targets['grasp_valid_l'] = 0

        is_valid = 1
        left_valid = float(data['left_bbox'] is not None)
        right_valid = float(data['right_bbox'] is not None)
        targets['grasp_valid_r'] = right_valid
        targets['grasp_valid_l'] = left_valid
        targets["is_valid"] = float(is_valid)
        targets["left_valid"] = float(left_valid) * targets['is_valid']
        targets["right_valid"] = float(right_valid) * targets['is_valid']
        targets["joints_valid_r"] = np.zeros(21) * targets['right_valid']
        targets["joints_valid_l"] = np.zeros(21) * targets['left_valid']

        if args.get('use_render_seg_loss', False):
            meta_info['is_mask_loss'] = 1
            targets['render.r'] = torch.from_numpy(mask_img_r[0:1]).float() # its a mask, all channels are same
            targets['render.l'] = torch.from_numpy(mask_img_l[0:1]).float() # its a mask, all channels are same
            targets['render_valid_r'] = float(modal_labels['right']==1) * left_valid
            targets['render_valid_l'] = float(modal_labels['left']==1) * right_valid

        if args.get('use_depth_loss', False):
            meta_info['is_depth_loss'] = 0
            targets['depth.r'] = torch.zeros((args.img_res, args.img_res))
            targets['depth.l'] = torch.zeros((args.img_res, args.img_res))

        return inputs, targets, meta_info