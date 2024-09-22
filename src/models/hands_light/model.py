import torch
import torch.nn as nn
import torch.nn.functional as F

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info, vit_conv
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead

import pytorch3d.transforms.rotation_conversions as rot_conv
from src.models.hands_light.renderer import MANORenderer


class HandsLight(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super(HandsLight, self).__init__()
        self.args = args
        if backbone == "resnet50":
            from src.nets.backbone.resnet import resnet50 as resnet
            self.backbone = resnet(pretrained=True)
        elif backbone == "resnet18":
            from src.nets.backbone.resnet import resnet18 as resnet
            self.backbone = resnet(pretrained=True)
        elif backbone == 'vit_b_16':
            from torchvision.models import vit_b_16 as vit
            self.backbone = vit(weights='DEFAULT')
            self.vit_conv = vit_conv()
            self.vit_spatial_dim = get_backbone_info(backbone)["spatial_dim"]
        else:
            assert False

        self.use_glb_feat = args.get('use_glb_feat', False)
        self.is_tf_decoder = args.get('tf_decoder', False)
        
        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3, tf_decoder=self.is_tf_decoder, args=args)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3, tf_decoder=self.is_tf_decoder, args=args)

        if args.separate_hands:
            if 'resnet' in backbone:
                self.hand_backbone_r = resnet(pretrained=True)
                self.hand_backbone_l = resnet(pretrained=True)
                conv1 = self.hand_backbone_r.conv1
            elif 'vit' in backbone:
                self.hand_backbone_r = vit(weights='DEFAULT')
                self.hand_backbone_r_vit_conv = vit_conv()
                self.hand_backbone_l = vit(weights='DEFAULT')
                self.hand_backbone_l_vit_conv = vit_conv()
                conv1 = self.hand_backbone_r.conv_proj
        else:
            if 'resnet' in backbone:
                self.hand_backbone = resnet(pretrained=True)
                conv1 = self.hand_backbone.conv1
            elif 'vit' in backbone:
                self.hand_backbone = vit(weights='DEFAULT')
                self.hand_backbone_vit_conv = vit_conv()
                conv1 = self.hand_backbone.conv_proj
        
        if args.pos_enc == 'center': inp_dim = 3 + 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'corner': inp_dim = 3 + 4 * 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'center+corner': inp_dim = 3 + 5 * 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'dense': inp_dim = 3 + 4 * args.n_freq_pos_enc
        else: inp_dim = 3
        
        if inp_dim != conv1.in_channels:
            if args.separate_hands:
                if 'resnet' in backbone:
                    self.hand_backbone_r.conv1 = nn.Conv2d(inp_dim, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
                    self.hand_backbone_l.conv1 = nn.Conv2d(inp_dim, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
                elif 'vit' in backbone:
                    self.hand_backbone_r.conv_proj = nn.Conv2d(inp_dim, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
                    self.hand_backbone_l.conv_proj = nn.Conv2d(inp_dim, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
            else:
                if 'resnet' in backbone:
                    self.hand_backbone.conv1 = nn.Conv2d(inp_dim, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
                elif 'vit' in backbone:
                    self.hand_backbone.conv_proj = nn.Conv2d(inp_dim, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding)

        if args.pos_enc == 'dense_latent':
            feat_conv_dim = feat_dim + 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'center+corner_latent':
            feat_conv_dim = feat_dim + 5 * 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'sinusoidal_cc':
            feat_conv_dim = feat_dim + 5 * 4 * args.n_freq_pos_enc
        elif args.pos_enc == 'cam_conv':
            feat_conv_dim = feat_dim + 6
        else:
            feat_conv_dim = feat_dim
        
        self.feature_conv = nn.Sequential(
            nn.Conv2d(feat_conv_dim, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, feat_dim),
            nn.ReLU(inplace=True),
        )

        self.mano_r = MANOHead(is_rhand=True, focal_length=focal_length, img_res=img_res)
        self.mano_l = MANOHead(is_rhand=False, focal_length=focal_length, img_res=img_res)

        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length

        self.use_grasp_loss = args.get('use_grasp_loss', False)
        if self.use_grasp_loss:
            # grasp type classification based on David Fouhey's NeurIPS 2023 paper 'Towards A Richer 2D Understanding of Hands at Scale'
            inp_grasp_dim = 10+16*3*3
            self.use_glb_feat_w_grasp = args.get('use_glb_feat_w_grasp', False)
            if self.use_glb_feat_w_grasp:
                inp_grasp_dim = 10+16*3*3+feat_dim
            self.grasp_classifier = nn.Sequential(
                nn.Linear(inp_grasp_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 9),
            )

        self.no_crops = args.get('no_crops', False)

        self.render_seg_loss = args.get('use_render_seg_loss', False)
        if self.render_seg_loss:
            self.renderer = MANORenderer(args)

        self.use_depth_loss = args.get('use_depth_loss', False)
        if self.use_depth_loss:
            self.init_grid() # 7x7 grid since feature resolution is 7x7
            self.depth_mlp = nn.Sequential(
                        nn.Conv2d(feat_conv_dim+2, 256, 3, 1, 1),
                        nn.ReLU(True),
                        nn.Conv2d(256, 256, 3, 1, 1),
                        nn.ReLU(True),
                        nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                        nn.Conv2d(256, 128, 3, 1, 1),
                        nn.ReLU(True),
                        nn.Conv2d(128, 128, 3, 1, 1),
                        nn.ReLU(True),
                        nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                        nn.Conv2d(128, 64, 3, 1, 1),
                        nn.ReLU(True),
                        nn.Conv2d(64, 32, 3, 1, 1),
                        nn.ReLU(True),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                        nn.Conv2d(32, 16, 3, 1, 1),
                        nn.ReLU(True),
                        nn.Conv2d(16, 1, 3, 1, 1),
                        )

        if args.regress_center_corner:
            # regress centers and corners
            self.corner_head = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4 * 2),
            )
            self.center_head = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2),
            )

    def init_grid(self):
        x = torch.linspace(-1, 1, 7)
        y = torch.linspace(-1, 1, 7)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)

    def broadcast(self, z):
        b = z.size(0)
        x_grid = self.x_grid.expand(b, 1, -1, -1).to(z.device)
        y_grid = self.y_grid.expand(b, 1, -1, -1).to(z.device)
        h_dim, w_dim = z.shape[2:]
        feat_grid = torch.cat((z, x_grid, y_grid), dim=1)
        return feat_grid

    def forward(self, inputs, meta_info):
        images = inputs["img"]
        K = meta_info["intrinsics"]

        if self.use_glb_feat:
            if 'resnet' in self.args.backbone:
                features = self.backbone(images)
            elif 'vit' in self.args.backbone:
                features = self.vit_forward(images, self.backbone, self.vit_conv)
            feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)

        bz, c, w, h = inputs["img"].shape
        if self.no_crops:
            r_features = features
            l_features = features
        else:
            if self.args.pos_enc == 'center':
                r_center_pos_enc = self.compute_center_pos_enc(inputs["r_center_angle"])
                l_center_pos_enc = self.compute_center_pos_enc(inputs["l_center_angle"])
                r_inp = torch.cat([inputs["r_img"], r_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)], dim=1)
                l_inp = torch.cat([inputs["l_img"], l_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)], dim=1)
            elif self.args.pos_enc == 'corner':
                r_corner_pos_enc = self.compute_corner_pos_enc(inputs["r_corner_angle"])
                l_corner_pos_enc = self.compute_corner_pos_enc(inputs["l_corner_angle"])
                r_inp = torch.cat([inputs["r_img"], r_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)], dim=1)
                l_inp = torch.cat([inputs["l_img"], l_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)], dim=1)
            elif self.args.pos_enc == 'center+corner':
                r_center_pos_enc = self.compute_center_pos_enc(inputs["r_center_angle"])
                l_center_pos_enc = self.compute_center_pos_enc(inputs["l_center_angle"])
                r_corner_pos_enc = self.compute_corner_pos_enc(inputs["r_corner_angle"])
                l_corner_pos_enc = self.compute_corner_pos_enc(inputs["l_corner_angle"])
                r_inp = torch.cat([inputs["r_img"], r_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h), r_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)], dim=1)
                l_inp = torch.cat([inputs["l_img"], l_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h), l_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)], dim=1)
            elif self.args.pos_enc == 'dense':
                r_dense_pos_enc = self.compute_dense_pos_enc(inputs["r_dense_angle"], inputs["r_dense_mask"])
                l_dense_pos_enc = self.compute_dense_pos_enc(inputs["l_dense_angle"], inputs["l_dense_mask"])
                r_inp = torch.cat([inputs["r_img"], r_dense_pos_enc], dim=1)
                l_inp = torch.cat([inputs["l_img"], l_dense_pos_enc], dim=1)
            else:
                r_inp = inputs["r_img"]
                l_inp = inputs["l_img"]
            
            if self.args.separate_hands:
                if 'resnet' in self.args.backbone:
                    r_features = self.hand_backbone_r(r_inp)
                    l_features = self.hand_backbone_l(l_inp)
                elif 'vit' in self.args.backbone:
                    r_features = self.vit_forward(r_inp, self.hand_backbone_r, self.hand_backbone_r_vit_conv)
                    l_features = self.vit_forward(l_inp, self.hand_backbone_l, self.hand_backbone_l_vit_conv)
            else:
                if 'resnet' in self.args.backbone:
                    r_features = self.hand_backbone(r_inp)
                    l_features = self.hand_backbone(l_inp)
                elif 'vit' in self.args.backbone:
                    r_features = self.vit_forward(r_inp, self.hand_backbone, self.hand_backbone_vit_conv)
                    l_features = self.vit_forward(l_inp, self.hand_backbone, self.hand_backbone_vit_conv)

            if self.args.pos_enc == 'dense_latent':
                r_dense_pos_enc = self.compute_dense_pos_enc(inputs["r_dense_angle"], inputs["r_dense_mask"])
                l_dense_pos_enc = self.compute_dense_pos_enc(inputs["l_dense_angle"], inputs["l_dense_mask"])
                w, h = r_features.shape[2:]
                r_dense_pos_enc = F.interpolate(r_dense_pos_enc, size=(w,h), mode='bilinear', align_corners=True)
                l_dense_pos_enc = F.interpolate(l_dense_pos_enc, size=(w,h), mode='bilinear', align_corners=True)
                
                if self.use_glb_feat:
                    r_features = torch.cat([r_features+features, r_dense_pos_enc], dim=1)
                    l_features = torch.cat([l_features+features, l_dense_pos_enc], dim=1)
                else:
                    r_features = torch.cat([r_features, r_dense_pos_enc], dim=1)
                    l_features = torch.cat([l_features, l_dense_pos_enc], dim=1)

            elif self.args.pos_enc == 'center+corner_latent':
                r_center_pos_enc = self.compute_center_pos_enc(inputs["r_center_angle"])
                l_center_pos_enc = self.compute_center_pos_enc(inputs["l_center_angle"])
                r_corner_pos_enc = self.compute_corner_pos_enc(inputs["r_corner_angle"])
                l_corner_pos_enc = self.compute_corner_pos_enc(inputs["l_corner_angle"])
                c, w, h = r_features.shape[1:]
                r_center_pos_enc = r_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
                r_corner_pos_enc = r_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
                l_center_pos_enc = l_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
                l_corner_pos_enc = l_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
                
                if self.use_glb_feat:
                    r_features = torch.cat([r_features+features, r_center_pos_enc, r_corner_pos_enc], dim=1)
                    l_features = torch.cat([l_features+features, l_center_pos_enc, l_corner_pos_enc], dim=1)
                else:
                    r_features = torch.cat([r_features, r_center_pos_enc, r_corner_pos_enc], dim=1)
                    l_features = torch.cat([l_features, l_center_pos_enc, l_corner_pos_enc], dim=1)

            elif self.args.pos_enc == 'cam_conv':
                r_dense_pos_enc = self.compute_cam_conv_pos_enc(inputs["r_dense_angle"], inputs["r_dense_mask"])
                l_dense_pos_enc = self.compute_cam_conv_pos_enc(inputs["l_dense_angle"], inputs["l_dense_mask"])
                c, w, h = r_features.shape[1:]
                r_dense_pos_enc = F.interpolate(r_dense_pos_enc, size=(w,h), mode='bilinear', align_corners=True)
                l_dense_pos_enc = F.interpolate(l_dense_pos_enc, size=(w,h), mode='bilinear', align_corners=True)
                
                if self.use_glb_feat:
                    r_features = torch.cat([r_features+features, r_dense_pos_enc], dim=1)
                    l_features = torch.cat([l_features+features, l_dense_pos_enc], dim=1)
                else:
                    r_features = torch.cat([r_features, r_dense_pos_enc], dim=1)
                    l_features = torch.cat([l_features, l_dense_pos_enc], dim=1)

            elif self.args.pos_enc == 'sinusoidal_cc':
                r_center_pos_enc = self.compute_center_pos_enc(inputs["r_center_angle"])
                l_center_pos_enc = self.compute_center_pos_enc(inputs["l_center_angle"])
                r_corner_pos_enc = self.compute_corner_pos_enc(inputs["r_corner_angle"])
                l_corner_pos_enc = self.compute_corner_pos_enc(inputs["l_corner_angle"])
                c, w, h = r_features.shape[1:]
                r_center_pos_enc = r_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
                l_center_pos_enc = l_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
                r_corner_pos_enc = r_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
                l_corner_pos_enc = l_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
                
                if self.use_glb_feat:
                    r_features = torch.cat([r_features+features, r_center_pos_enc, r_corner_pos_enc], dim=1)
                    l_features = torch.cat([l_features+features, l_center_pos_enc, l_corner_pos_enc], dim=1)
                else:
                    r_features = torch.cat([r_features, r_center_pos_enc, r_corner_pos_enc], dim=1)
                    l_features = torch.cat([l_features, l_center_pos_enc, l_corner_pos_enc], dim=1)

            if self.use_depth_loss:
                depth_r = self.predict_depth(r_features)
                depth_l = self.predict_depth(l_features)

            if not self.is_tf_decoder:
                r_features = self.feature_conv(r_features)
                l_features = self.feature_conv(l_features)

        if self.no_crops:
            hmr_output_r = self.head_r(features)
            hmr_output_l = self.head_l(features)
        else:
            hmr_output_r = self.head_r(r_features, use_pool=False)
            hmr_output_l = self.head_l(l_features, use_pool=False)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]
        root_l = hmr_output_l["cam_t.wp"]

        root_r_init = hmr_output_r["cam_t.wp.init"]
        root_l_init = hmr_output_l["cam_t.wp.init"]

        if self.args.pos_enc == 'pcl':
            glb_rot_r = hmr_output_r["pose"][:,0]
            hmr_output_r["pose"][:,0] = torch.bmm(inputs['r_rot'], glb_rot_r)
            glb_rot_l = hmr_output_l["pose"][:,0]
            hmr_output_l["pose"][:,0] = torch.bmm(inputs['l_rot'], glb_rot_l)

        pose_r = hmr_output_r["pose"]
        shape_r = hmr_output_r["shape"]
        pose_l = hmr_output_l["pose"]
        shape_l = hmr_output_l["shape"]

        if sum(meta_info['is_flipped']) > 0: # verify this
            flip_root_r = hmr_output_l["cam_t.wp"] * torch.Tensor([[1, -1, 1]]).to(root_r.device)
            flip_root_l = hmr_output_r["cam_t.wp"] * torch.Tensor([[1, -1, 1]]).to(root_l.device)
            
            pose_r_axis_angle = rot_conv.matrix_to_axis_angle(hmr_output_l['pose']).view(bz,-1)
            pose_r_axis_angle[:, 1::3] *= -1
            pose_r_axis_angle[:, 2::3] *= -1
            flip_pose_r = rot_conv.axis_angle_to_matrix(pose_r_axis_angle.view(bz,pose_r.shape[1],-1)).view(bz,pose_r.shape[1],3,3)

            pose_l_axis_angle = rot_conv.matrix_to_axis_angle(hmr_output_r['pose']).view(bz,-1)
            pose_l_axis_angle[:, 1::3] *= -1
            pose_l_axis_angle[:, 2::3] *= -1
            flip_pose_l = rot_conv.axis_angle_to_matrix(pose_l_axis_angle.view(bz,pose_l.shape[1],-1)).view(bz,pose_l.shape[1],3,3)

            flip_shape_r = hmr_output_l["shape"]
            flip_shape_l = hmr_output_r["shape"]

            flip_root_r_init = hmr_output_l["cam_t.wp.init"] * torch.Tensor([[1, -1, 1]]).to(root_r_init.device)
            flip_root_l_init = hmr_output_r["cam_t.wp.init"] * torch.Tensor([[1, -1, 1]]).to(root_l_init.device)

            root_r = torch.where(meta_info['is_flipped'].bool().unsqueeze(1).repeat(1,root_r.shape[1]), flip_root_r, root_r)
            root_l = torch.where(meta_info['is_flipped'].bool().unsqueeze(1).repeat(1,root_l.shape[1]), flip_root_l, root_l)
            pose_r = torch.where(meta_info['is_flipped'].bool().unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,pose_r.shape[1],pose_r.shape[2],pose_r.shape[3]), flip_pose_r, pose_r)
            pose_l = torch.where(meta_info['is_flipped'].bool().unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,pose_l.shape[1],pose_l.shape[2],pose_l.shape[3]), flip_pose_l, pose_l)
            shape_r = torch.where(meta_info['is_flipped'].bool().unsqueeze(1).repeat(1,shape_r.shape[1]), flip_shape_r, shape_r)
            shape_l = torch.where(meta_info['is_flipped'].bool().unsqueeze(1).repeat(1,shape_l.shape[1]), flip_shape_l, shape_l)
            root_r_init = torch.where(meta_info['is_flipped'].bool().unsqueeze(1).repeat(1,root_r_init.shape[1]), flip_root_r_init, root_r_init)
            root_l_init = torch.where(meta_info['is_flipped'].bool().unsqueeze(1).repeat(1,root_l_init.shape[1]), flip_root_l_init, root_l_init)

        if self.args.pos_enc == 'perspective_correction':
            r_angle = torch.cat([-inputs['r_center_angle'], torch.zeros((bz,1)).to(K.device)], dim=-1) # [B, 2] yaw & pitch
            r_rotmat = rot_conv.euler_angles_to_matrix(r_angle, convention='XYZ')
            pose_r[:, 0] = torch.matmul(r_rotmat, pose_r[: ,0])
            l_angle = torch.cat([-inputs['l_center_angle'], torch.zeros((bz,1)).to(K.device)], dim=-1) # [B, 2] yaw & pitch
            l_rotmat = rot_conv.euler_angles_to_matrix(l_angle, convention='XYZ')
            pose_l[:, 0] = torch.matmul(l_rotmat, pose_l[: ,0])

        mano_output_r = self.mano_r(
            rotmat=pose_r,
            shape=shape_r,
            K=K,
            cam=root_r,
        )

        mano_output_l = self.mano_l(
            rotmat=pose_l,
            shape=shape_l,
            K=K,
            cam=root_l,
        )

        mano_output_r["cam_t.wp.init.r"] = root_r_init
        mano_output_l["cam_t.wp.init.l"] = root_l_init

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)

        if self.use_grasp_loss:
            if self.use_glb_feat_w_grasp:
                right_grasp = self.grasp_classifier(torch.cat([hmr_output_r['shape'], hmr_output_r['pose'].view(bz,-1), feat_vec], dim=1))
                left_grasp = self.grasp_classifier(torch.cat([hmr_output_l['shape'], hmr_output_l['pose'].view(bz,-1), feat_vec], dim=1))
            else:
                right_grasp = self.grasp_classifier(torch.cat([hmr_output_r['shape'], hmr_output_r['pose'].view(bz,-1)], dim=1))
                left_grasp = self.grasp_classifier(torch.cat([hmr_output_l['shape'], hmr_output_l['pose'].view(bz,-1)], dim=1))
            grasp_output = xdict()
            grasp_output['grasp.r'] = right_grasp
            grasp_output['grasp.l'] = left_grasp
            output.merge(grasp_output)

        if self.render_seg_loss:
            # render seg masks from MANO mesh
            render_r = self.renderer(mano_output_r, meta_info, is_right=True)
            render_l = self.renderer(mano_output_l, meta_info, is_right=False)
            render_output = xdict()
            render_output['render.r'] = render_r['mask']
            render_output['render.l'] = render_l['mask']
            output.merge(render_output)

        if self.use_depth_loss:
            depth_output = xdict()
            depth_output['depth.r'] = depth_r.squeeze(1)
            depth_output['depth.l'] = depth_l.squeeze(1)
            output.merge(depth_output)

        if self.args.regress_center_corner:
            # regress centers and corners
            angle_output = xdict()
            angle_output['center.r'] = self.center_head(r_features)
            angle_output['center.l'] = self.center_head(l_features)
            angle_output['corner.r'] = self.corner_head(r_features)
            angle_output['corner.l'] = self.corner_head(l_features)
            output.merge(angle_output)
        
        return output

    def predict_depth(self, feat):
        z = self.broadcast(feat)
        x = self.depth_mlp(z)
        return x

    def compute_center_pos_enc(self, angle):
        # center positional encoding for all pixels
        L = self.args.n_freq_pos_enc
        bz, c = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c)
        center_pos_enc = torch.stack([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=-1).reshape(bz,-1).float()
        return center_pos_enc

    def compute_corner_pos_enc(self, angle):
        # corner positional encoding for all pixels
        L = self.args.n_freq_pos_enc
        bz, c = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c)
        corner_pos_enc = torch.stack([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=-1).reshape(bz,-1).float()
        return corner_pos_enc

    def compute_dense_pos_enc(self, angle, mask):
        # dense positional encoding for all pixels
        L = self.args.n_freq_pos_enc
        bz, c, w, h = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1,1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c,w,h)
        dense_pos_enc = torch.cat([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=3).reshape(bz, -1, w, h).float()
        mask_expand = mask.unsqueeze(1).repeat(1,2*L*c,1,1)
        dense_pos_enc = dense_pos_enc * mask_expand
        dense_pos_enc = F.interpolate(dense_pos_enc, size=(self.args.img_res_ds, self.args.img_res_ds), mode='bilinear', align_corners=True)
        return dense_pos_enc

    def compute_cam_conv_pos_enc(self, angle, mask):
        # dense positional encoding for all pixels
        L = self.args.n_freq_pos_enc
        bz, c, w, h = angle.shape
        mask = mask.unsqueeze(1).repeat(1,c,1,1)
        angle = angle * mask
        dense_pos_enc = F.interpolate(angle, size=(self.args.img_res_ds, self.args.img_res_ds), mode='bilinear', align_corners=True)
        return dense_pos_enc

    def vit_forward(self, images, net, net_conv):
        conv_feat = net._process_input(images)
        bz = conv_feat.shape[0]
        batch_class_token =  net.class_token.expand(bz, -1, -1)
        x = torch.cat([batch_class_token, conv_feat], dim=1)
        x = net.encoder(x)
        feat = x[:, 1:]
        spatial_feat = feat.permute(0,2,1).reshape(bz, -1, self.vit_spatial_dim, self.vit_spatial_dim) # bz, 768, 14, 14
        features = net_conv(spatial_feat) # bz, 2048, 7, 7

        return features