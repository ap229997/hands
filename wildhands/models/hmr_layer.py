import torch
import torch.nn as nn
# from wildhands.models.transformer import *


class HMRLayer(nn.Module):
    def __init__(self, feat_dim, mid_dim, specs_dict, **kwargs):
        super().__init__()

        self.feat_dim = feat_dim
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.specs_dict = specs_dict

        vector_dim = sum(list(zip(*specs_dict.items()))[1])
        hmr_dim = feat_dim + vector_dim

        # self.tf_decoder = kwargs.get('tf_decoder', False)
        # if self.tf_decoder:
        #     tf_decoder_layer = TransformerDecoderLayer(d_model=mid_dim, nhead=1, dim_feedforward=mid_dim, batch_first=True)
        #     self.vector_mlp = nn.Sequential(
        #         nn.Linear(1, mid_dim), # vector_dim used instead of 1 in old tf decoder block
        #         nn.ReLU(),
        #         # nn.Dropout(),
        #     )
        #     args = kwargs.get('args', None)
        #     inp_feat_dim = feat_dim
        #     if args is not None:
        #         if args.pos_enc == 'center+corner_latent':
        #             inp_feat_dim = inp_feat_dim + 5 * 4 * args.n_freq_pos_enc
        #         elif args.pos_enc == 'dense_latent':
        #             inp_feat_dim = inp_feat_dim + 4 * args.n_freq_pos_enc
            
        #     self.feat_mlp = nn.Sequential(
        #         nn.Linear(inp_feat_dim, mid_dim),
        #         nn.ReLU(),
        #         # nn.Dropout(),
        #     )
        #     self.refine_decoder = TransformerDecoder(decoder_layer=tf_decoder_layer, num_layers=1)
        #     self.refine_dropout = nn.Dropout()

        #     tf_encoder_layer = TransformerEncoderLayer(d_model=mid_dim, nhead=1, dim_feedforward=mid_dim, batch_first=True)
        #     self.self_attn = TransformerEncoder(encoder_layer=tf_encoder_layer, num_layers=1)
        # else:
        # construct refine
        self.refine = nn.Sequential(
            nn.Linear(hmr_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(),
        )

        # construct decoders
        decoders = {}
        for key, vec_size in specs_dict.items():
            decoders[key] = nn.Linear(mid_dim, vec_size)
        self.decoders = nn.ModuleDict(decoders)

        self.init_weights()

    def init_weights(self):
        for key, decoder in self.decoders.items():
            nn.init.xavier_uniform_(decoder.weight, gain=0.01)
            self.decoders[key] = decoder

    def forward(self, feat, init_vector_dict, n_iter):
        pred_vector_dict = init_vector_dict
        for i in range(n_iter):
            vectors = list(zip(*pred_vector_dict.items()))[1]
            # if self.tf_decoder:
            #     ####### this block is likely to be wrong #######
            #     # tgt = torch.cat(list(vectors), dim=1)
            #     # memory = self.feat_mlp(feat)
            #     # tgt = self.vector_mlp(tgt)
            #     ################################################

            #     tgt = torch.cat(list(vectors), dim=1).unsqueeze(-1)
            #     tgt = self.vector_mlp(tgt)
            #     # memory = self.feat_mlp(feat).unsqueeze(1) # single token only
            #     bz, c, h, w = feat.shape
            #     memory = self.feat_mlp(feat.view(bz, c, -1).permute(0, 2, 1))
            #     xc = self.refine_decoder(tgt, memory, no_norm=True)
            #     xc = self.self_attn(xc, no_norm=True) # self attention on the target tokens
            #     xc = torch.mean(xc, dim=1)
            #     # xc = self.refine_dropout(xc)
            # else:
            xc = torch.cat([feat] + list(vectors), dim=1)
            xc = self.refine(xc)
            for key, decoder in self.decoders.items():
                pred_vector_dict.overwrite(key, decoder(xc) + pred_vector_dict[key])

        pred_vector_dict.has_invalid()
        return pred_vector_dict
