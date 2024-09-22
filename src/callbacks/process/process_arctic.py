import common.camera as camera


def process_data_light(
    models, inputs, targets, meta_info, mode, args, field_max=float("inf")
):
    img_res = args.img_res
    K = meta_info["intrinsics"]
    gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
    gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

    gt_pose_l = targets["mano.pose.l"]  # MANO pose parameters
    gt_betas_l = targets["mano.beta.l"]  # MANO beta parameters

    # pose MANO in MANO canonical space
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints # MANO canonical space
    gt_vertices_r = gt_out_r.vertices # MANO canonical space
    gt_root_cano_r = gt_out_r.joints[:, 0] # MANO canonical space

    targets['mano.joints3d.r'] = gt_out_r.joints # MANO canonical space
    targets['mano.vertices.r'] = gt_out_r.vertices # MANO canonical space

    gt_out_l = models["mano_l"](
        betas=gt_betas_l,
        hand_pose=gt_pose_l[:, 3:],
        global_orient=gt_pose_l[:, :3],
        transl=None,
    )
    gt_model_joints_l = gt_out_l.joints # MANO canonical space
    gt_vertices_l = gt_out_l.vertices # MANO canonical space
    gt_root_cano_l = gt_out_l.joints[:, 0] # MANO canonical space

    targets['mano.joints3d.l'] = gt_out_l.joints # MANO canonical space
    targets['mano.vertices.l'] = gt_out_l.vertices # MANO canonical space

    # translation from MANO cano space to camera coord space
    Tr0 = (targets['mano.j3d.full.r'] - gt_model_joints_r).mean(dim=1)
    Tl0 = (targets['mano.j3d.full.l']- gt_model_joints_l).mean(dim=1)
    gt_vertices_r = gt_vertices_r + Tr0[:, None, :]
    gt_vertices_l = gt_vertices_l + Tl0[:, None, :]

    # roots
    gt_root_cam_patch_r = targets['mano.j3d.full.r'][:, 0]
    gt_root_cam_patch_l = targets['mano.j3d.full.l'][:, 0]
    gt_cam_t_r = gt_root_cam_patch_r - gt_root_cano_r
    gt_cam_t_l = gt_root_cam_patch_l - gt_root_cano_l
    # gt_cam_t_o = gt_transl

    targets["mano.cam_t.r"] = gt_cam_t_r
    targets["mano.cam_t.l"] = gt_cam_t_l

    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    gt_cam_t_wp_r = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_r, avg_focal_length, img_res
    )

    gt_cam_t_wp_l = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_l, avg_focal_length, img_res
    )

    targets["mano.cam_t.wp.r"] = gt_cam_t_wp_r
    targets["mano.cam_t.wp.l"] = gt_cam_t_wp_l

    targets["mano.v3d.cam.r"] = gt_vertices_r
    targets["mano.v3d.cam.l"] = gt_vertices_l
    targets["mano.j3d.cam.r"] = targets['mano.j3d.full.r']
    targets["mano.j3d.cam.l"] = targets['mano.j3d.full.l']

    return inputs, targets, meta_info
