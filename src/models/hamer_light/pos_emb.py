import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, args, feat_dim, patch_size):
        super(PositionalEncoding, self).__init__()
        
        self.args = args
        if self.args.pos_enc == 'dense_latent':
            inp_dim = 4 * args.n_freq_pos_enc
        elif self.args.pos_enc == 'center+corner_latent':
            inp_dim = 5 * 4 * args.n_freq_pos_enc
        else:
            raise ValueError(f"Unsupported positional encoding type: {self.args.pos_enc} for {self.args.method.split('_')[0]}")

        self.patch_size = patch_size
        self.feat_dim = feat_dim

        self.feat_mlp = nn.Sequential(
            nn.Linear(inp_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pos_emb, prefix='r_'):
        
        if self.args.pos_enc == 'center+corner_latent':
            bz = pos_emb[prefix+"center_angle"].shape[0]
            center_pos_enc = self.compute_center_pos_enc(pos_emb[prefix+"center_angle"])
            corner_pos_enc = self.compute_corner_pos_enc(pos_emb[prefix+"corner_angle"])
            tf_h, tf_w = self.patch_size
            tf_pos_enc = torch.cat([center_pos_enc, corner_pos_enc], dim=1)
            tf_pos_enc = self.feat_mlp(tf_pos_enc)
            tf_pos_enc = tf_pos_enc.view(bz,1,self.feat_dim).repeat(1,tf_h*tf_w,1)

        elif self.args.pos_enc == 'dense_latent':
            bz = pos_emb[prefix+"dense_angle"].shape[0]
            dense_pos_enc = self.compute_dense_pos_enc(pos_emb[prefix+"dense_angle"], pos_emb[prefix+"dense_mask"])
            tf_h, tf_w = self.patch_size
            tf_pos_enc = torch.nn.functional.interpolate(dense_pos_enc, size=(tf_h,tf_w), mode='bilinear', align_corners=True)
            tf_pos_enc = self.feat_mlp(tf_pos_enc).permute(0,2,3,1).view(bz,-1,self.feat_dim)
        
        return tf_pos_enc.float()

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

    def compute_dense_pos_enc(self, angle, mask, size):
        # dense positional encoding for all pixels
        L = self.args.n_freq_pos_enc
        bz, c, w, h = angle.shape
        freq_expand = 2**torch.arange(L).unsqueeze(0).repeat(bz,1).reshape(bz,-1,1,1,1).to(angle.device)
        angle_expand = angle.reshape(bz,1,c,w,h)
        dense_pos_enc = torch.cat([torch.sin(freq_expand*angle_expand), torch.cos(freq_expand*angle_expand)], dim=3).reshape(bz, -1, w, h).float()
        mask_expand = mask.unsqueeze(1).repeat(1,2*L*c,1,1)
        dense_pos_enc = dense_pos_enc * mask_expand
        dense_pos_enc = torch.nn.functional.interpolate(dense_pos_enc, size=size, mode='bilinear', align_corners=True)
        return dense_pos_enc