from src.parsers.configs.generic import DEFAULT_ARGS_EGO

DEFAULT_ARGS_EGO["batch_size"] = 32
DEFAULT_ARGS_EGO["test_batch_size"] = 32
DEFAULT_ARGS_EGO["num_workers"] = 8
DEFAULT_ARGS_EGO["use_gt_bbox"] = True
DEFAULT_ARGS_EGO["separate_hands"] = False
DEFAULT_ARGS_EGO["pos_enc"] = None # no positional encoding
DEFAULT_ARGS_EGO["n_freq_pos_enc"] = 4
DEFAULT_ARGS_EGO["img_res"] = 224
DEFAULT_ARGS_EGO["img_res_ds"] = 224
DEFAULT_ARGS_EGO["logger"] = 'tensorboard'
DEFAULT_ARGS_EGO['backbone'] = 'resnet50'
DEFAULT_ARGS_EGO['vis_every'] = 100
DEFAULT_ARGS_EGO['log_every'] = 50
DEFAULT_ARGS_EGO['regress_center_corner'] = False
DEFAULT_ARGS_EGO['flip_prob'] = 0.0
DEFAULT_ARGS_EGO['dataset'] = 'hands+assembly+epic_grasp+epic_seg'
DEFAULT_ARGS_EGO['val_dataset'] = 'epic'
DEFAULT_ARGS_EGO['tf_decoder'] = False
DEFAULT_ARGS_EGO['use_glb_feat'] = True
DEFAULT_ARGS_EGO['use_grasp_loss'] = True
DEFAULT_ARGS_EGO['use_glb_feat_w_grasp'] = False
DEFAULT_ARGS_EGO['use_render_seg_loss'] = True
DEFAULT_ARGS_EGO['use_gt_hand_mask'] = False
DEFAULT_ARGS_EGO['use_depth_loss'] = False
DEFAULT_ARGS_EGO['no_crops'] = True
DEFAULT_ARGS_EGO['eval_every_epoch'] = 1
DEFAULT_ARGS_EGO['no_intrx'] = False