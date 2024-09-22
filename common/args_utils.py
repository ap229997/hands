from loguru import logger


def set_default_params(args, default_args):
    # if a val is not set on argparse, use default val
    # else, use the one in the argparse
    custom_dict = {}
    for key, val in args.items():
        if val is None:
            args[key] = default_args[key]
        else:
            custom_dict[key] = val

    logger.info(f"Using custom values: {custom_dict}")
    return args


def set_extra_params(args, default_args):
    # use new keys from default_args
    custom_dict = {}
    for key, val in default_args.items():
        if key not in args:
            args[key] = val
            custom_dict[key] = val

    logger.info(f"Using default values: {custom_dict}")
    return args