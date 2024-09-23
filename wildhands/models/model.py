import torch
import torch.nn as nn

from wildhands.common.xdict import xdict
from wildhands.models.backbone.utils import get_backbone_info
from wildhands.models.hand_heads.hand_hmr import HandHMR
from wildhands.models.hand_heads.mano_head import MANOHead

class WildHands(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super().__init__()
        self.args = args
        from wildhands.models.backbone.resnet import resnet50 as resnet
        self.backbone = resnet(pretrained=True)

        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3, args=args)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3, args=args)

        self.hand_backbone = resnet(pretrained=True)
        
        feat_conv_dim = feat_dim + 5 * 4 * args.n_freq_pos_enc
        
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

        self.img_res = img_res
        self.focal_length = focal_length

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

        features = self.backbone(images) # global features help as well

        bz, c, w, h = inputs["img"].shape
        
        r_inp = inputs["r_img"]
        l_inp = inputs["l_img"]
        
        r_features = self.hand_backbone(r_inp)
        l_features = self.hand_backbone(l_inp)

        r_center_pos_enc = self.compute_center_pos_enc(inputs["r_center_angle"])
        l_center_pos_enc = self.compute_center_pos_enc(inputs["l_center_angle"])
        r_corner_pos_enc = self.compute_corner_pos_enc(inputs["r_corner_angle"])
        l_corner_pos_enc = self.compute_corner_pos_enc(inputs["l_corner_angle"])
        c, w, h = r_features.shape[1:]
        r_center_pos_enc = r_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
        r_corner_pos_enc = r_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
        l_center_pos_enc = l_center_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
        l_corner_pos_enc = l_corner_pos_enc.view(bz,-1,1,1).repeat(1,1,w,h)
        
        r_features = torch.cat([r_features+features, r_center_pos_enc, r_corner_pos_enc], dim=1)
        l_features = torch.cat([l_features+features, l_center_pos_enc, l_corner_pos_enc], dim=1)

        r_features = self.feature_conv(r_features)
        l_features = self.feature_conv(l_features)

        hmr_output_r = self.head_r(r_features, use_pool=False)
        hmr_output_l = self.head_l(l_features, use_pool=False)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]
        root_l = hmr_output_l["cam_t.wp"]
        root_r_init = hmr_output_r["cam_t.wp.init"]
        root_l_init = hmr_output_l["cam_t.wp.init"]

        # poses
        pose_r = hmr_output_r["pose"]
        shape_r = hmr_output_r["shape"]
        pose_l = hmr_output_l["pose"]
        shape_l = hmr_output_l["shape"]

        # MANO outputs
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
        
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        
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