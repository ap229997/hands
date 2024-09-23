from easydict import EasyDict
import wildhands.configs.model_config as config

def construct_args():
    default_args = config.DEFAULT_ARGS_EGO
    args = EasyDict(default_args)

    args.focal_length = 1000.0 # this work for EPIC images
    args.img_norm_mean = [0.485, 0.456, 0.406]
    args.img_norm_std = [0.229, 0.224, 0.225]

    return args
