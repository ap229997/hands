import os, sys, json, pickle
from types import FrameType
from cv2 import inpaint
from loguru import logger
import numpy as np
from PIL import Image, ImageDraw
from torch.nn.modules.container import T
from tqdm import tqdm
import torch, torchvision
from torch.utils.data import DataLoader
sys.path.append(".")

from common.xdict import xdict

from common.torch_utils import reset_all_seeds
from common.data_utils import denormalize_images
import common.transforms as tf
import src.factory as factory
from src.datasets.sample_dataset import SampleDataset
from common.body_models import build_mano_aa


def batch_to_device(batch, device):
    inputs, targets, meta_info = batch
    inputs = {k: v.to(device) for k, v in inputs.items()}
    targets = {k: v.to(device) for k, v in targets.items()}
    for k, v in meta_info.items():
        if isinstance(v, torch.Tensor):
            meta_info[k] = v.to(device)
    return inputs, targets, meta_info

def move_to_cpu(curr_dict):
    new_dict = {}
    for k, v in curr_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to("cpu")
        else:
            new_dict[k] = v
    return new_dict

def draw_joints(draw, joints, color='red', size=5):
    for j in joints:
        draw.ellipse((j[0]-size//2, j[1]-size//2, j[0]+size//2, j[1]+size//2), outline=color)
    return draw

def main():
    reset_all_seeds(1)

    args = xdict()
    args.img_res = 224
    args.img_norm_mean = [0.485, 0.456, 0.406]
    args.img_norm_std = [0.229, 0.224, 0.225]
    args.setup = 'p2' # specific to arctic
    args.speedup = True # specific to arctic
    args.ego_image_scale = 0.3 # specific to arctic, use 1.0 for other datasets
    args.focal_length = 1000.0
    args.rot_factor = 30.0 # predefined values, use as is
    args.noise_factor = 0.4 # predefined values, use as is
    args.scale_factor = 0.25 # predefined values, use as is
    args.flip_prob = 0.0 # predefined values, use as is
    args.use_gt_k = True

    mano_r = build_mano_aa(is_rhand=True)
    mano_l = build_mano_aa(is_rhand=False)

    dataset = SampleDataset(args=args, split='val')
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=2,
            num_workers=0,
            shuffle=True,
        )

    args.vis = True
    save_dir = 'logs/sample_data'
    if args.vis:
        os.makedirs(save_dir, exist_ok=True)

    cnt = 0
    for idx, batch in enumerate(tqdm(dataloader)):
        inputs, targets, meta_info = batch

        gt_r = targets['mano.j2d.norm.r'] # 2d joints
        gt_l = targets['mano.j2d.norm.l']

        intrx = meta_info['intrinsics']

        gt_3d_r = targets['mano.j3d.full.r']
        gt_3d_l = targets['mano.j3d.full.l']
        j3d_pix_r = tf.project2d_batch(intrx, gt_3d_r) # projection into the image
        j3d_pix_l = tf.project2d_batch(intrx, gt_3d_l) # projection into the image

        valid_r = targets['joints_valid_r']
        valid_l = targets['joints_valid_l']

        gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
        gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

        gt_pose_l = targets["mano.pose.l"]  # MANO pose parameters
        gt_betas_l = targets["mano.beta.l"]  # MANO beta parameters

        # pose MANO in MANO canonical space
        gt_out_r = mano_r(
            betas=gt_betas_r,
            hand_pose=gt_pose_r[:, 3:],
            global_orient=gt_pose_r[:, :3],
            transl=None,
        )
        gt_model_joints_r = gt_out_r.joints # MANO canonical space
        gt_vertices_r = gt_out_r.vertices # MANO canonical space

        gt_out_l = mano_l(
            betas=gt_betas_l,
            hand_pose=gt_pose_l[:, 3:],
            global_orient=gt_pose_l[:, :3],
            transl=None,
        )
        gt_model_joints_l = gt_out_l.joints # MANO canonical space
        gt_vertices_l = gt_out_l.vertices # MANO canonical space

        # translation from MANO cano space to camera coord space
        Tr0 = (targets['mano.j3d.full.r'] - gt_model_joints_r).mean(dim=1)
        Tl0 = (targets['mano.j3d.full.l']- gt_model_joints_l).mean(dim=1)
        gt_v_r = gt_vertices_r + Tr0[:, None, :] # (B, 778, 3)
        gt_v_l = gt_vertices_l + Tl0[:, None, :] # (B, 778, 3)
        
        v_pix_r = tf.project2d_batch(intrx, gt_v_r)
        v_pix_l = tf.project2d_batch(intrx, gt_v_l)
        
        images = denormalize_images(inputs['img'].cpu())
        bz = gt_r.shape[0]
        for i in range(bz):
            g_r = gt_r[i].numpy()
            v_r = valid_r[i].numpy()
            g_l = gt_l[i].numpy()
            v_l = valid_l[i].numpy()

            g_r = ((g_r+1)/2)*args.img_res
            g_l = ((g_l+1)/2)*args.img_res

            if args.vis and (cnt+1) % 1 == 0:
                img = images[i].numpy().transpose(1, 2, 0)
                pil_og_img = Image.fromarray(np.uint8(img*255))
                pil_img = Image.fromarray(np.uint8(img*255))
                pil_img_3d = Image.fromarray(np.uint8(img*255))
                pil_img_hand = Image.fromarray(np.uint8(img*255))
                draw = ImageDraw.Draw(pil_img)
                draw_3d = ImageDraw.Draw(pil_img_3d)
                draw_hand = ImageDraw.Draw(pil_img_hand)

                if targets['right_valid'][i] > 0:
                    draw = draw_joints(draw, g_r[v_r>0], color='green')
                    draw_3d = draw_joints(draw_3d, j3d_pix_r[i].numpy(), color='blue')
                    draw_hand = draw_joints(draw_hand, v_pix_r[i].numpy(), color='red')
                if targets['left_valid'][i] > 0:
                    draw = draw_joints(draw, g_l[v_l>0], color='cyan')
                    draw_3d = draw_joints(draw_3d, j3d_pix_l[i].numpy(), color='orange')
                    draw_hand = draw_joints(draw_hand, v_pix_l[i].numpy(), color='yellow')

                # concatenate PIL images
                save_img = np.concatenate([np.array(pil_og_img), np.array(pil_img), np.array(pil_img_3d), np.array(pil_img_hand)], axis=1)
                name = meta_info['imgname'][i].split('/')[-1]
                Image.fromarray(save_img).save(os.path.join(save_dir, name))

            cnt += 1
        
        if cnt == 100:
            break


if __name__ == "__main__":
    main()
