import torch
import torch.nn as nn
from torch.nn import functional as F
from .hand_head import hand_regHead, hand_Encoder
from .mano_head import mano_regHead

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.hand_regHead = hand_regHead()
        self.hand_Encoder = hand_Encoder()
        self.mano_regHead = mano_regHead()
    
    def forward(self, feats):
        out_hm, encoding, preds_joints_img = self.hand_regHead(feats)
        mano_encoding = self.hand_Encoder(out_hm, encoding)
        pred_mano_results = self.mano_regHead(mano_encoding)

        return pred_mano_results
