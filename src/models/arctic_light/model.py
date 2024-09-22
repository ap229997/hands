from src.models.hands_light.model import HandsLight


class ArcticSFLight(HandsLight):
    def __init__(self, backbone, focal_length, img_res, args):
        super().__init__(backbone, focal_length, img_res, args)

    def forward(self, inputs, meta_info):
        return super().forward(inputs, meta_info)