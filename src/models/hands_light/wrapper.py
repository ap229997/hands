from src.models.hands_light.model import HandsLight
from src.models.generic.wrapper import GenericWrapper


def mul_loss_dict(loss_dict):
    for key, val in loss_dict.items():
        loss, weight = val
        loss_dict[key] = loss * weight
    return loss_dict

class HandsWrapper(GenericWrapper):
    def __init__(self, args, push_images_fn):
        super().__init__(args, push_images_fn)
        self.model = HandsLight(
            backbone=args.backbone,
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )

    def inference(self, inputs, meta_info):
        return super().inference_pose(inputs, meta_info)

    def forward(self, inputs, targets, meta_info, mode):
        return super().forward(inputs, targets, meta_info, mode)
