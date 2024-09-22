from src.models.hamer_light.model import HAMER
from src.models.generic.wrapper import GenericWrapper


class HaMeRWrapper(GenericWrapper):
    def __init__(self, args, push_images_fn):
        super().__init__(args, push_images_fn)
        self.model = HAMER(
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )

    def inference(self, inputs, meta_info):
        return super().inference_pose(inputs, meta_info)

    def forward(self, inputs, targets, meta_info, mode):
        return super().forward(inputs, targets, meta_info, mode)
