from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
import cv2, math
import os
import os.path as osp
from scipy import linalg
from copy import deepcopy

import common.data_utils as data_utils
from src.datasets.dataset_utils import pad_jts2d, downsample
import json
from pycocotools.coco import COCO

ANNOT_VERSION = "v1-1"
N_DEBUG_SAMPLES = 400


def load_skeleton(path, joint_num):
    # load joint_world info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


def world2cam(pts_3d, R, t):
    pts_cam = np.dot(R, pts_3d.T).T + t
    return pts_cam

def cam2world(pts_cam_3d, R, t):
    inv_R = np.linalg.inv(R)
    pts_3d = np.dot(inv_R, (pts_cam_3d - t).T).T
    return pts_3d


class Camera(object):
    def __init__(self, K, Rt, dist=None, name=""):
        # Rotate first then translate
        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.Rt = np.array(Rt).copy()
        assert self.Rt.shape == (3, 4)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return np.dot(self.K, self.Rt)

    def factor(self):
        """  Factorize the camera matrix into K,R,t as P = K[R|t]. """

        # factor first 3*3 part
        K,R = linalg.rq(self.projection[:, :3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        K = np.dot(K,T)
        R = np.dot(T,R) # T is its own inverse
        t = np.dot(linalg.inv(self.K), self.projection[:,3])

        return K, R, t

    def get_params(self):
        K, R, t = self.factor()
        campos, camrot = t, R
        focal = [K[0, 0], K[1, 1]]
        princpt = [K[0, 2], K[1, 2]]
        return campos, camrot, focal, princpt


class AssemblyDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode.replace('mini','').replace('tiny','').replace('small','')  # train, test, val
        self.img_path = f"{os.environ['DATA_DIR']}/assembly/images"
        self.annot_path = f"{os.environ['DATA_DIR']}/assembly/annotations"
        self.modality = "ego"
        self.transform = transforms.Compose([
                transforms.ToTensor()
            ]) if 'train' in self.mode else transforms.ToTensor()
        self.normalize_img = transforms.Normalize(mean=args.img_norm_mean, std=args.img_norm_std)
        self.args = args
        self.joint_num = 21
        self.root_joint_idx = {"right": 20, "left": 41}
        # assembly to mano joint mapping
        self.joint_type = {
            "right": np.array([20, 7, 6, 5, 11, 10, 9, 19, 18, 17, 15, 14, 13, 3, 2, 1, 0, 4, 8, 12, 16]),
            "left": np.array([41, 28, 27, 26, 32, 31, 30, 40, 39, 38, 36, 35, 34, 24, 23, 22, 21, 25, 29, 33, 37]),
        }
        # mano to assembly joint mapping
        self.joint_type_inv = {'right': [], 'left': []}
        hand_ids = self.joint_type['right']
        for i, j in enumerate(hand_ids):
            self.joint_type_inv['right'].append(np.where(hand_ids == i)[0][0])
        self.joint_type_inv['right'] = np.array(self.joint_type_inv['right'])
        self.joint_type_inv['left'] = self.joint_type_inv['right'].copy() # concatenate right and left hand preds
        
        self.skeleton = load_skeleton(
            osp.join(self.annot_path, "skeleton.txt"), self.joint_num * 2
        )

        self.is_debug = args.debug
        self.aug_data = 'train' in self.mode

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        n_skip = 0
        # load annotation
        print(f"Load annotation from  {self.annot_path}, mode: {self.mode}")
        data_mode = self.mode
        self.invalid_data_file = os.path.join(
            self.annot_path, data_mode, f"invalid_{data_mode}_{self.modality}.txt"
        )
        db = COCO(
            osp.join(
                self.annot_path,
                data_mode,
                "assemblyhands_"
                + data_mode
                + f"_{self.modality}_data_{ANNOT_VERSION}.json",
            )
        )
        with open(
            osp.join(
                self.annot_path,
                data_mode,
                "assemblyhands_"
                + data_mode
                + f"_{self.modality}_calib_{ANNOT_VERSION}.json",
            )
        ) as f:
            cameras = json.load(f)["calibration"]
        with open(
            osp.join(
                self.annot_path,
                data_mode,
                "assemblyhands_" + data_mode + f"_joint_3d_{ANNOT_VERSION}.json",
            )
        ) as f:
            joints = json.load(f)["annotations"]

        print("Get bbox and root depth from groundtruth annotation")

        annot_list = db.anns.keys()
        for i, aid in enumerate(annot_list):
            ann = db.anns[aid]
            image_id = ann["image_id"]
            img = db.loadImgs(image_id)[0]

            seq_name = str(img["seq_name"])
            camera_name = img["camera"]
            frame_idx = img["frame_idx"]
            file_name = img["file_name"]
            img_path = osp.join(self.img_path, file_name)
            assert osp.exists(img_path), f"Image path {img_path} does not exist"

            K = np.array(
                cameras[seq_name]["intrinsics"][camera_name + "_mono10bit"],
                dtype=np.float32,
            )
            Rt = np.array(
                cameras[seq_name]["extrinsics"][f"{frame_idx:06d}"][
                    camera_name + "_mono10bit"
                ],
                dtype=np.float32,
            )
            retval_camera = Camera(K, Rt, dist=None, name=camera_name)
            campos, camrot, focal, princpt = retval_camera.get_params()

            joint_world = np.array(
                joints[seq_name][f"{frame_idx:06d}"]["world_coord"], dtype=np.float32
            )
            joint_cam = world2cam(joint_world, camrot, campos)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

            joint_valid = np.array(ann["joint_valid"], dtype=np.float32).reshape(
                self.joint_num * 2
            )

            abs_depth = {
                "right": joint_cam[self.root_joint_idx["right"], 2],
                "left": joint_cam[self.root_joint_idx["left"], 2],
            }
            cam_param = {"focal": focal, "princpt": princpt}

            # bbox in [x0, y0, x1, y1] format
            right_bbox = deepcopy(ann["bbox"]["right"])
            left_bbox = deepcopy(ann["bbox"]["left"])
            # convert to [x0, y0, w, h] format to be consistent with other datasets
            if right_bbox is not None:
                right_bbox = [right_bbox[0], right_bbox[1], right_bbox[2]-right_bbox[0], right_bbox[3]-right_bbox[1]]
            if left_bbox is not None:
                left_bbox = [left_bbox[0], left_bbox[1], left_bbox[2]-left_bbox[0], left_bbox[3]-left_bbox[1]]

            joint = {
                "cam_coord": joint_cam,
                "img_coord": joint_img,
                "world_coord": joint_world,
                "valid": joint_valid,
            }  # joint_valid}
            data = {
                "img_path": img_path,
                "img_width": img["width"],
                "img_height": img["height"],
                "seq_name": seq_name,
                "cam_param": cam_param,
                "right_bbox": right_bbox,
                "left_bbox": left_bbox,
                "joint": joint,
                "abs_depth": abs_depth,
                "file_name": img["file_name"],
                "seq_name": seq_name,
                "cam": camera_name,
                "frame": frame_idx,
                "retval_camera": retval_camera,
            }
            self.datalist.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

            if self.is_debug and i >= N_DEBUG_SAMPLES - 1:
                print(">>> DEBUG MODE: Loaded %d samples" % N_DEBUG_SAMPLES)
                break
        
        assert len(self.datalist) > 0, "No data found."

        # subsample indices
        all_keys = list(range(len(self.datalist)))
        self.subsampled_keys = downsample(all_keys, mode)
        logger.info("# samples in Assembly %s: %d" % (mode, len(self.subsampled_keys)))

    def handtype_str2array(self, hand_type):
        if hand_type == "right":
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == "left":
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == "interacting":
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print("Not supported hand type: " + hand_type)

    def __len__(self):
        if self.args.debug:
            return 1
        return len(self.subsampled_keys)

    def __getitem__(self, index):
        idx = self.subsampled_keys[index]
        args = self.args
        data = self.datalist[idx]
        img_path, right_bbox, left_bbox, joint = (
            data["img_path"],
            data["right_bbox"],
            data["left_bbox"],
            data["joint"],
        )
        joint_world = joint["world_coord"].copy()
        joint_cam = joint["cam_coord"].copy()
        joint_img = joint["img_coord"].copy()
        joint_valid = joint["valid"].copy()

        data_cam = joint_cam
        data_2d = joint_img

        intrx = data['retval_camera'].K.copy()

        joints2d_r = pad_jts2d(data_2d[self.joint_type['right']])
        joints3d_r = data_cam[self.joint_type['right']] / 1000 # mm -> m

        joints2d_l = pad_jts2d(data_2d[self.joint_type['left']])
        joints3d_l = data_cam[self.joint_type['left']] / 1000 # mm -> m

        bbox = [data['img_width']//2, data['img_height']//2, max(data['img_width'], data['img_height'])/200.0]
        is_egocam = True

        cv_img, img_status = data_utils.read_img(img_path, (data['img_width'], data['img_height']))

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )

        img = data_utils.rgb_processing(
            self.aug_data,
            cv_img,
            center,
            scale,
            augm_dict,
            img_res=args.img_res,
        )

        right_bbox = data['right_bbox']
        left_bbox = data['left_bbox']
        if right_bbox is not None:
            end_pts = np.array([[right_bbox[0], right_bbox[1]], [right_bbox[0]+right_bbox[2], right_bbox[1]+right_bbox[3]]])
            end_pts = data_utils.j2d_processing(pad_jts2d(end_pts), center, scale, augm_dict, args.img_res)
            end_pts = ((end_pts[...,:2]+1)/2)*args.img_res
            end_pts = end_pts.flatten()
            right_bbox = [end_pts[0], end_pts[1], end_pts[2]-end_pts[0], end_pts[3]-end_pts[1]]
            right_bbox = np.array(right_bbox).astype(np.int16)
            right_bbox_og = right_bbox.copy()
        else:
            right_bbox_og = np.array([0, 0, args.img_res-1, args.img_res-1])
        if left_bbox is not None:
            end_pts = np.array([[left_bbox[0], left_bbox[1]], [left_bbox[0]+left_bbox[2], left_bbox[1]+left_bbox[3]]])
            end_pts = data_utils.j2d_processing(pad_jts2d(end_pts), center, scale, augm_dict, args.img_res)
            end_pts = ((end_pts[...,:2]+1)/2)*args.img_res
            end_pts = end_pts.flatten()
            left_bbox = [end_pts[0], end_pts[1], end_pts[2]-end_pts[0], end_pts[3]-end_pts[1]]
            left_bbox = np.array(left_bbox).astype(np.int16)
            left_bbox_og = left_bbox.copy()
        else:
            left_bbox_og = np.array([0, 0, args.img_res-1, args.img_res-1])

        if self.aug_data:
            right_bbox, left_bbox = data_utils.jitter_bbox(right_bbox), data_utils.jitter_bbox(left_bbox)
            if right_bbox is not None:
                new_right_bbox = np.array([right_bbox[0], right_bbox[1], right_bbox[0]+right_bbox[2], right_bbox[1]+right_bbox[3]]).astype(np.int16).clip(0, args.img_res-1)
                if (new_right_bbox[2]-new_right_bbox[0]) == 0 or (new_right_bbox[3]-new_right_bbox[1]) == 0: right_bbox = None
            if left_bbox is not None:
                new_left_bbox = np.array([left_bbox[0], left_bbox[1], left_bbox[0]+left_bbox[2], left_bbox[1]+left_bbox[3]]).astype(np.int16).clip(0, args.img_res-1)
                if (new_left_bbox[2]-new_left_bbox[0]) == 0 or (new_left_bbox[3]-new_left_bbox[1]) == 0: left_bbox = None
        
        bbox_scale = 1.75
        if args.get('bbox_scale', None) is not None:
            bbox_scale = args.bbox_scale
        # bbox format below; [x0, y0, x1, y1] for r_bbox, l_bbox
        r_img, r_bbox = data_utils.crop_and_pad(img, right_bbox, args, scale=bbox_scale)
        l_img, l_bbox = data_utils.crop_and_pad(img, left_bbox, args, scale=bbox_scale)
        norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
        norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

        img_ds = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [args.img_res/2, args.img_res/2, args.img_res, args.img_res], 
                            1.0, 0.0, [args.img_res_ds, args.img_res_ds], cv2.INTER_CUBIC)[0].transpose(2,0,1)
        img_ds = np.clip(img_ds, 0, 1)
        img_ds = torch.from_numpy(img_ds).float()
        norm_img = self.normalize_img(img_ds)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        inputs["r_img"] = norm_r_img
        inputs["l_img"] = norm_l_img
        inputs["r_bbox"] = r_bbox
        inputs["l_bbox"] = l_bbox

        if augm_dict["flip"] == 1:
            inputs['img'] = torch.flip(norm_img, dims=[2])
            inputs['r_img'] = torch.flip(norm_r_img, dims=[2])
            inputs['l_img'] = torch.flip(norm_l_img, dims=[2])
            n_r_bbox = np.array([args.img_res-1, 0, args.img_res-1, 0]) + np.array([-1, 1, -1, 1]) * l_bbox
            n_l_bbox = np.array([args.img_res-1, 0, args.img_res-1, 0]) + np.array([-1, 1, -1, 1]) * r_bbox
            inputs['r_bbox'] = np.array([n_r_bbox[2], n_r_bbox[1], n_r_bbox[0], n_r_bbox[3]])
            inputs['l_bbox'] = np.array([n_l_bbox[2], n_l_bbox[1], n_l_bbox[0], n_l_bbox[3]])

        if args.use_gt_bbox:
            inputs['r_bbox_og'] = right_bbox_og
            inputs['l_bbox_og'] = left_bbox_og

        meta_info["imgname"] = img_path

        # hands
        targets["mano.pose.r"] = torch.zeros((48,)) # dummy values
        targets["mano.pose.l"] = torch.zeros((48,)) # dummy values
        targets["mano.beta.r"] = torch.tensor([0.82747316,  0.13775729, -0.39435294, 0.17889787, -0.73901576, 0.7788163, -0.5702684, 0.4947751, -0.24890041, 1.5943261]) # dummy values = mean value of beta from the dataset, shouldn't matter
        targets["mano.beta.l"] = torch.tensor([-0.19330633, -0.08867972, -2.5790455, -0.10344583, -0.71684015, -0.28285977, 0.55171007, -0.8403888, -0.8490544, -1.3397144])
        targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets["mano.j2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()

        targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        targets["mano.j3d.full.l"] = torch.FloatTensor(joints3d_l[:, :3])

        # scale and center in the original image space
        scale_original = max([data["img_width"], data["img_height"]]) / 200.0
        center_original = [data["img_width"] / 2.0, data['img_height'] / 2.0]
        fixed_focal_length = args.focal_length
        if args.get('no_intrx', False):
            fixed_focal_length = 1.0
        intrx = data_utils.get_aug_intrix(
            intrx,
            fixed_focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        # TODO: abstract this to a function which can be used across different encodings and datasets
        if args.pos_enc is not None: # compute positional encoding for different pixels
            L = args.n_freq_pos_enc
            
            if 'center' in args.pos_enc or 'perspective_correction' in args.pos_enc:
                # center of left & right image
                r_center = (r_bbox[:2] + r_bbox[2:]) / 2.0
                l_center = (l_bbox[:2] + l_bbox[2:]) / 2.0
                r_angle_x, r_angle_y = np.arctan2(r_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(r_center[1]-intrx[1,2], intrx[1,1])
                l_angle_x, l_angle_y = np.arctan2(l_center[0]-intrx[0,2], intrx[0,0]), np.arctan2(l_center[1]-intrx[1,2], intrx[1,1])
                r_angle = np.array([r_angle_x, r_angle_y]).astype(np.float32)
                l_angle = np.array([l_angle_x, l_angle_y]).astype(np.float32)
                inputs["r_center_angle"], inputs["l_center_angle"] = r_angle, l_angle
                targets['center.r'], targets['center.l'] = r_angle, l_angle

            if 'corner' in args.pos_enc:
                # corners of left & right image
                r_corner = np.array([[r_bbox[0], r_bbox[1]], [r_bbox[0], r_bbox[3]], [r_bbox[2], r_bbox[1]], [r_bbox[2], r_bbox[3]]])
                l_corner = np.array([[l_bbox[0], l_bbox[1]], [l_bbox[0], l_bbox[3]], [l_bbox[2], l_bbox[1]], [l_bbox[2], l_bbox[3]]])
                r_corner = np.stack([r_corner[:,0]-intrx[0,2], r_corner[:,1]-intrx[1,2]], axis=-1)
                l_corner = np.stack([l_corner[:,0]-intrx[0,2], l_corner[:,1]-intrx[1,2]], axis=-1)
                r_angle = np.arctan2(r_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                l_angle = np.arctan2(l_corner, np.array([[intrx[0,0], intrx[1,1]]])).flatten().astype(np.float32)
                inputs["r_corner_angle"], inputs["l_corner_angle"] = r_angle, l_angle
                targets['corner.r'], targets['corner.l'] = r_angle, l_angle

            if 'dense' in args.pos_enc:
                # dense positional encoding for all pixels
                r_x_grid, r_y_grid = range(r_bbox[0], r_bbox[2]+1), range(r_bbox[1], r_bbox[3]+1)
                r_x_grid, r_y_grid = np.meshgrid(r_x_grid, r_y_grid, indexing='ij') # torch doesn't support batched meshgrid
                l_x_grid, l_y_grid = range(l_bbox[0], l_bbox[2]+1), range(l_bbox[1], l_bbox[3]+1)
                l_x_grid, l_y_grid = np.meshgrid(l_x_grid, l_y_grid, indexing='ij')
                r_pix = np.stack([r_x_grid-intrx[0,2], r_y_grid-intrx[1,2]], axis=-1)
                l_pix = np.stack([l_x_grid-intrx[0,2], l_y_grid-intrx[1,2]], axis=-1)
                r_angle = np.arctan2(r_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)
                l_angle = np.arctan2(l_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)
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
                r_pix = np.stack([r_x_grid-intrx[0,2], r_y_grid-intrx[1,2]], axis=-1)
                l_pix = np.stack([l_x_grid-intrx[0,2], l_y_grid-intrx[1,2]], axis=-1)
                r_pix_transp = r_pix.transpose(2,0,1).astype(np.float32)
                l_pix_transp = l_pix.transpose(2,0,1).astype(np.float32)
                r_pix_centered = np.stack([2*r_x_grid/args.img_res-1, 2*r_y_grid/args.img_res-1], axis=-1).transpose(2,0,1).astype(np.float32)
                l_pix_centered = np.stack([2*l_x_grid/args.img_res-1, 2*l_y_grid/args.img_res-1], axis=-1).transpose(2,0,1).astype(np.float32)
                r_angle = np.arctan2(r_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)
                l_angle = np.arctan2(l_pix, np.array([[intrx[0,0], intrx[1,1]]])).transpose(2,0,1).astype(np.float32)
                
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
        meta_info["dist"] = torch.FloatTensor(torch.zeros(8)) # dummy value
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        meta_info['dataset'] = 'assembly'

        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info['is_j2d_loss'] = 1
        meta_info['is_j3d_loss'] = 1
        meta_info['is_beta_loss'] = 0
        meta_info['is_pose_loss'] = 0
        meta_info['is_cam_loss'] = 0

        meta_info['is_grasp_loss'] = 0
        targets["grasp.l"] = 8 # no grasp
        targets["grasp.r"] = 8 # no grasp
        targets['grasp_valid_r'] = 0
        targets['grasp_valid_l'] = 0

        # root and at least 3 joints inside image
        targets["is_valid"] = 1
        targets["left_valid"] = int(left_bbox is not None)
        targets["right_valid"] = int(right_bbox is not None)
        targets["joints_valid_r"] = joint_valid[self.joint_type["right"]]
        targets["joints_valid_l"] = joint_valid[self.joint_type["left"]]

        if args.get('use_render_seg_loss', False):
            # dummy values
            meta_info['is_mask_loss'] = 0
            targets['render.r'] = torch.zeros((1, args.img_res_ds, args.img_res_ds))
            targets['render.l'] = torch.zeros((1, args.img_res_ds, args.img_res_ds))
            targets['render_valid_r'] = 0
            targets['render_valid_l'] = 0

        if args.get('use_depth_loss', False):
            meta_info['is_depth_loss'] = 0
            targets['depth.r'] = torch.zeros((args.img_res, args.img_res))
            targets['depth.l'] = torch.zeros((args.img_res, args.img_res))

        if 'test' in self.mode:
            campos, camrot, _, _ = data['retval_camera'].get_params()
            meta_info['campos'] = torch.FloatTensor(campos)
            meta_info['camrot'] = torch.FloatTensor(camrot)
        
        return inputs, targets, meta_info