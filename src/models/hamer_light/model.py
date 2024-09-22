import os
from sys import prefix
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import vit
from .mano_head import MANOTransformerDecoderHead
from .pos_emb import PositionalEncoding

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead

from src.models.hands_light.renderer import MANORenderer

class HAMER(nn.Module):

    def __init__(self, args, focal_length, img_res):
        super().__init__()

        self.args = args

        # Create backbone feature extractor
        self.vit_input_size = (256, 192) # for loading pretrained weights
        self.backbone = vit(args, img_size=self.vit_input_size)

        # Create MANO head
        self.mano_head = MANOTransformerDecoderHead(args)

        # Load pretrained weights
        pretrained_ckpt = args.get('pretrained', 'vit')
        if pretrained_ckpt == 'vit':
            pretrained_backbone_weight = f"{os.environ['DATA_DIR']}/hamer_training_data/vitpose_backbone.pth"
            self.backbone.load_state_dict(torch.load(pretrained_backbone_weight, map_location='cpu')['state_dict'], strict=False)
        elif pretrained_ckpt == 'hamer':
            ckpt_path = f"{os.environ['DATA_DIR']}/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
            ckpt_weights = torch.load(ckpt_path, map_location='cpu')['state_dict']
            backbone_weights = {k.replace('backbone.', ''): v for k, v in ckpt_weights.items() if 'backbone' in k}
            mano_head_weights = {k.replace('mano_head.', ''): v for k, v in ckpt_weights.items() if 'mano_head' in k}
            self.backbone.load_state_dict(backbone_weights)
            self.mano_head.load_state_dict(mano_head_weights)

        # Instantiate MANO model
        self.mano_r = MANOHead(is_rhand=True, focal_length=focal_length, img_res=img_res)
        self.mano_l = MANOHead(is_rhand=False, focal_length=focal_length, img_res=img_res)

        # Setup KPE positional encoding
        self.pos_enc = args.get('pos_enc', None)
        if self.pos_enc is not None:
            self.kpe = PositionalEncoding(args, feat_dim=self.backbone.patch_embed.proj.out_channels, patch_size=self.backbone.patch_embed.patch_shape)

        # Auxiliary grasp supervision
        self.use_grasp_loss = args.get('use_grasp_loss', False)
        if self.use_grasp_loss:
            # 9-way classification based on David Fouhey's NeurIPS 2023 paper
            inp_grasp_dim = 10+16*3*3
            self.grasp_classifier = nn.Sequential(
                nn.Linear(inp_grasp_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 9),
            )

        # Rendered segmentation loss
        self.render_seg_loss = args.get('use_render_seg_loss', False)
        if self.render_seg_loss:
            self.renderer = MANORenderer(args)

    def forward(self, inputs, meta_info, targets=None):
        K = meta_info["intrinsics"]
        
        # Use right and left images as input
        r_inp = inputs['r_img']
        l_inp = inputs['l_img']
        # resize both to self.vit_input_size
        r_inp_resize = F.interpolate(r_inp, size=max(self.vit_input_size), mode='bilinear', align_corners=False)
        l_inp_resize = F.interpolate(l_inp, size=max(self.vit_input_size), mode='bilinear', align_corners=False)
        bz = r_inp.shape[0]
        
        x = torch.cat([r_inp_resize, l_inp_resize], dim=0)
        
        more_args = {}
        use_kpe_dec = False
        # compute KPE embedding
        if self.pos_enc is not None:
            r_kpe_emb = self.kpe(inputs, prefix='r_')
            l_kpe_emb = self.kpe(inputs, prefix='l_')
            kpe_emb = torch.cat([r_kpe_emb, l_kpe_emb], dim=0)
            more_args['kpe_emb'] = kpe_emb
            use_kpe_dec = True

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        conditioning_feats = self.backbone(x[:,:,:,32:-32], **more_args)

        if use_kpe_dec:
            kpe_reshape = kpe_emb.permute(0,2,1).reshape(x.shape[0], -1, conditioning_feats.shape[-2], conditioning_feats.shape[-1])
            conditioning_feats = conditioning_feats + kpe_reshape

        # Compute MANO parameters
        pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats)

        # split mano params for left and right hands
        theta_r = pred_mano_params['hand_pose'][:bz]
        global_orient_r = pred_mano_params['global_orient'][:bz]
        pose_r = torch.cat([global_orient_r, theta_r], dim=1)
        shape_r = pred_mano_params['betas'][:bz]
        root_r = pred_cam[:bz]

        theta_l = pred_mano_params['hand_pose'][bz:]
        global_orient_l = pred_mano_params['global_orient'][bz:]
        pose_l = torch.cat([global_orient_l, theta_l], dim=1)
        shape_l = pred_mano_params['betas'][bz:]
        root_l = pred_cam[bz:]

        mano_output_r = self.mano_r(rotmat=pose_r, shape=shape_r, K=K, cam=root_r)
        mano_output_l = self.mano_l(rotmat=pose_l, shape=shape_l, K=K, cam=root_l)

        mano_output_r["cam_t.wp.init.r"] = root_r
        mano_output_l["cam_t.wp.init.l"] = root_l

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)

        if self.use_grasp_loss:
            right_grasp = self.grasp_classifier(torch.cat([shape_r, pose_r.view(bz,-1)], dim=1))
            left_grasp = self.grasp_classifier(torch.cat([shape_l, pose_l.view(bz,-1)], dim=1))
            grasp_output = xdict()
            grasp_output['grasp.r'] = right_grasp
            grasp_output['grasp.l'] = left_grasp
            output.merge(grasp_output)

        if self.render_seg_loss:
            render_r = self.renderer(mano_output_r, meta_info, is_right=True)
            render_l = self.renderer(mano_output_l, meta_info, is_right=False)
            render_output = xdict()
            render_output['render.r'] = render_r['mask']
            render_output['render.l'] = render_l['mask']
            output.merge(render_output)

        return output