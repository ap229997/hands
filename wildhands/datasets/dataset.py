import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import wildhands.common.data_utils as data_utils


class WildHandsDataset(Dataset):
    def __init__(self,
                 args,
                 img: np.array,
                 boxes: np.array,
                 right: np.array,
                 focal_length: float,
                 rescale_factor: float = 1.5, # check what value to use
                 **kwargs):
        super().__init__()
        self.args = args
        self.mode = 'test'
        self.aug_data = False

        self.img = img
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)
        self.boxes = boxes.astype(np.float32)
        self.rescale_factor = rescale_factor
        self.focal_length = focal_length
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.right = right.astype(np.float32)

    def __len__(self):
        return len(self.personid)
    
    def __getitem__(self, idx):
        args = self.args
        image_size = {"width": self.img.shape[1], "height": self.img.shape[0]}
        augm_dict={"rot": 0, "sc": 1.0, "flip": 0, "pn": np.ones(3)}
        is_right = self.right[idx]
        center = [image_size['width'] / 2, image_size['height'] / 2]
        scale =  max(image_size['width'], image_size['height']) / 200

        img = data_utils.rgb_processing(
                self.aug_data,
                self.img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )

        if is_right:
            right_bbox = self.boxes[idx].astype(np.float32)
            end_pts = np.array([[right_bbox[0], right_bbox[1]], [right_bbox[2], right_bbox[3]]])
            end_pts = data_utils.j2d_processing(data_utils.pad_jts2d(end_pts), center, scale, augm_dict, args.img_res)
            end_pts = ((end_pts[...,:2]+1)/2)*args.img_res
            end_pts = end_pts.flatten().clip(0, args.img_res-1)
            right_bbox = [end_pts[0], end_pts[1], end_pts[2]-end_pts[0], end_pts[3]-end_pts[1]]
            left_bbox = None
        else:
            left_bbox = self.boxes[idx].astype(np.float32)
            end_pts = np.array([[left_bbox[0], left_bbox[1]], [left_bbox[2], left_bbox[3]]])
            end_pts = data_utils.j2d_processing(data_utils.pad_jts2d(end_pts), center, scale, augm_dict, args.img_res)
            end_pts = ((end_pts[...,:2]+1)/2)*args.img_res
            end_pts = end_pts.flatten().clip(0, args.img_res-1)
            left_bbox = [end_pts[0], end_pts[1], end_pts[2]-end_pts[0], end_pts[3]-end_pts[1]]
            right_bbox = None

        def crop_and_pad(img, bbox, scale=1):
            if bbox is None: # resize the image and return
                img_crop = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [args.img_res/2, args.img_res/2, args.img_res, args.img_res], 
                        1.0, 0.0, [args.img_res, args.img_res], cv2.INTER_CUBIC)[0].transpose(2,0,1)
                img_crop = np.clip(img_crop, 0, 1)
                new_bbox = np.array([0, 0, args.img_res-1, args.img_res-1])
                return img_crop, new_bbox

            x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
            x_mid, y_mid, width, height = (x0+x1)//2, (y0+y1)//2, x1-x0, y1-y0 
            size = max(width, height)
            img_crop = data_utils.generate_patch_image_clean(img.transpose(1,2,0), [x_mid, y_mid, size*scale, size*scale], 1.0, 0.0, [args.img_res, args.img_res], cv2.INTER_CUBIC)[0]
            img_crop = np.clip(img_crop, 0, 1)
            new_bbox = np.array([x_mid-(size*scale)//2, y_mid-(size*scale)//2, x_mid+(size*scale)//2, y_mid+(size*scale)//2]).clip(0, args.img_res-1).astype(np.int16)
            return img_crop.transpose(2,0,1), new_bbox

        r_img, r_bbox = crop_and_pad(img, right_bbox, scale=self.rescale_factor)
        l_img, l_bbox = crop_and_pad(img, left_bbox, scale=self.rescale_factor)
        norm_r_img = self.normalize_img(torch.from_numpy(r_img).float())
        norm_l_img = self.normalize_img(torch.from_numpy(l_img).float())

        img = torch.from_numpy(img).float()
        norm_img = self.normalize_img(img)

        inputs = {}
        meta_info = {}
        
        inputs["img"] = norm_img
        inputs["r_img"] = norm_r_img
        inputs["l_img"] = norm_l_img
        inputs["r_bbox"] = r_bbox
        inputs["l_bbox"] = l_bbox

        meta_info['right'] = is_right
        meta_info['person_id'] = self.personid[idx]

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = np.zeros((3, 3)) # dummy intrinsic value, default focal length = 1000
        fixed_focal_length = self.focal_length * (args.img_res / max(image_size["width"], image_size["height"]))
        intrx = data_utils.get_aug_intrix(
            intrx,
            fixed_focal_length,
            args.img_res,
            False,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if not isinstance(intrx, np.ndarray):
            intrx = intrx.numpy()
        meta_info["intrinsics"] = torch.FloatTensor(intrx)

        intrx_for_enc = intrx.copy()
        if args.get('no_intrx', False):
            intrx_for_enc = np.eye(3)
            intrx_for_enc[0,0] = args.img_res / 2
            intrx_for_enc[1,1] = args.img_res / 2
            intrx_for_enc[0,2] = args.img_res / 2
            intrx_for_enc[1,2] = args.img_res / 2

        args.pos_enc_decoder = args.get('pos_enc_decoder', None)
        if args.pos_enc is not None or args.pos_enc_decoder is not None: # compute positional encoding for different pixels
            # center of left & right image
            r_center = (r_bbox[:2] + r_bbox[2:]) / 2.0
            l_center = (l_bbox[:2] + l_bbox[2:]) / 2.0
            r_angle_x, r_angle_y = np.arctan2(r_center[0]-intrx_for_enc[0,2], intrx_for_enc[0,0]), np.arctan2(r_center[1]-intrx_for_enc[1,2], intrx_for_enc[1,1])
            l_angle_x, l_angle_y = np.arctan2(l_center[0]-intrx_for_enc[0,2], intrx_for_enc[0,0]), np.arctan2(l_center[1]-intrx_for_enc[1,2], intrx_for_enc[1,1])
            r_angle = np.array([r_angle_x, r_angle_y]).astype(np.float32)
            l_angle = np.array([l_angle_x, l_angle_y]).astype(np.float32)
            inputs["r_center_angle"], inputs["l_center_angle"] = r_angle, l_angle

            # corners of left & right image
            r_corner = np.array([[r_bbox[0], r_bbox[1]], [r_bbox[0], r_bbox[3]], [r_bbox[2], r_bbox[1]], [r_bbox[2], r_bbox[3]]])
            l_corner = np.array([[l_bbox[0], l_bbox[1]], [l_bbox[0], l_bbox[3]], [l_bbox[2], l_bbox[1]], [l_bbox[2], l_bbox[3]]])
            r_corner = np.stack([r_corner[:,0]-intrx_for_enc[0,2], r_corner[:,1]-intrx_for_enc[1,2]], axis=-1)
            l_corner = np.stack([l_corner[:,0]-intrx_for_enc[0,2], l_corner[:,1]-intrx_for_enc[1,2]], axis=-1)
            r_angle = np.arctan2(r_corner, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).flatten().astype(np.float32)
            l_angle = np.arctan2(l_corner, np.array([[intrx_for_enc[0,0], intrx_for_enc[1,1]]])).flatten().astype(np.float32)
            inputs["r_corner_angle"], inputs["l_corner_angle"] = r_angle, l_angle

        return inputs, meta_info