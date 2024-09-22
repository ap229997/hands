import torch
import torch.nn as nn
from torch.nn import functional as F
from .backbone import FPN
from .transformer import Transformer
from .regressor import Regressor

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead

from src.models.hamer_light.pos_emb import PositionalEncoding
from src.models.hands_light.renderer import MANORenderer


class HandOccNet(nn.Module):
    def __init__(self, focal_length, img_res, args):
        super().__init__()

        self.input_size = (256, 256)
        self.backbone = FPN(pretrained=True)
        self.FIT = Transformer(injection=True) # feature injecting transformer
        self.SET = Transformer(injection=False) # self enhancing transformer
        self.regressor = Regressor()
        
        self.FIT.apply(init_weights)
        self.SET.apply(init_weights)
        self.regressor.apply(init_weights)

        # Instantiate MANO model
        self.mano_r = MANOHead(is_rhand=True, focal_length=focal_length, img_res=img_res)
        self.mano_l = MANOHead(is_rhand=False, focal_length=focal_length, img_res=img_res)

        # Setup KPE positional encoding
        self.pos_enc = args.get('pos_enc', None)
        if args.pos_enc is not None:
            self.kpe = PositionalEncoding(args, feat_dim=self.backbone.latlayer3.out_channels, patch_size=(32, 32))

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
        # resize both to self.input_size
        r_inp_resize = F.interpolate(r_inp, size=max(self.input_size), mode='bilinear', align_corners=False)
        l_inp_resize = F.interpolate(l_inp, size=max(self.input_size), mode='bilinear', align_corners=False)
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

        p_feats, s_feats = self.backbone(x)
        feats = self.FIT(s_feats, p_feats, **more_args)
        feats = self.SET(feats, feats, **more_args)

        if use_kpe_dec:
            feats = feats + kpe_emb.permute(0,2,1).view(feats.shape)
        pred_mano_results = self.regressor(feats)

        # split mano params for left and right hands
        pose_r = pred_mano_results['mano_pose'][:bz]
        shape_r = pred_mano_results['mano_shape'][:bz]
        root_r = pred_mano_results['cam'][:bz]

        pose_l = pred_mano_results['mano_pose'][bz:]
        shape_l = pred_mano_results['mano_shape'][bz:]
        root_l = pred_mano_results['cam'][bz:]

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

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)
