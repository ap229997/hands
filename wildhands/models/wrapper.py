import pytorch_lightning as pl

from wildhands.common.xdict import xdict
from wildhands.models.model import WildHands
from wildhands.models.hand_heads.mano_head import build_mano_aa


class WildHandsWrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mano_r = build_mano_aa(is_rhand=True)
        self.mano_l = build_mano_aa(is_rhand=False)
        self.add_module("mano_r", self.mano_r)
        self.add_module("mano_l", self.mano_l)

        self.model = WildHands(
            backbone=args.backbone,
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )

    def forward(self, batch):
        inputs, meta_info = batch
        return self.inference_pose(inputs, meta_info)

    def inference_pose(self, inputs, meta_info):
        pred = self.model(inputs, meta_info)
        mydict = xdict()
        mydict.merge(xdict(inputs).prefix("inputs."))
        mydict.merge(pred.prefix("pred."))
        mydict.merge(xdict(meta_info).prefix("meta_info."))
        mydict = mydict.detach()
        return mydict
