import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import common.thing as thing
import common.transforms as tf
import common.vis_utils as vis_utils
from common.data_utils import denormalize_images
from common.mesh import Mesh
from common.rend_utils import color2material
from common.torch_utils import unpad_vtensor

mesh_color_dict = {
    "right": [100,100,254], # blue, og: [200, 200, 250], yufei: [183,216,254]
    "left": [183,100,254], # purple, og: [100, 100, 250], yufei: [183,216,254]
    "object": [144, 250, 100],
    "top": [144, 250, 100],
    "bottom": [129, 159, 214],
}


def visualize_one_example(
    images_i,
    kp2d_proj_b_i,
    kp2d_proj_t_i,
    joints2d_r_i,
    joints2d_l_i,
    kp2d_b_i,
    kp2d_t_i,
    bbox2d_b_i,
    bbox2d_t_i,
    joints2d_proj_r_i,
    joints2d_proj_l_i,
    bbox2d_proj_b_i,
    bbox2d_proj_t_i,
    joints_valid_r,
    joints_valid_l,
    flag,
    only_hands
):
    # whether the hand is cleary visible
    valid_idx_r = (joints_valid_r.long() == 1).nonzero().view(-1).numpy()
    valid_idx_l = (joints_valid_l.long() == 1).nonzero().view(-1).numpy()

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.reshape(-1)

    # GT 2d keypoints (good overlap as it is from perspective camera)
    ax[0].imshow(images_i)
    if not only_hands:
        ax[0].scatter(
            kp2d_b_i[:, 0], kp2d_b_i[:, 1], color="r"
        )  # keypoints from bottom part of object
        ax[0].scatter(kp2d_t_i[:, 0], kp2d_t_i[:, 1], color="b")  # keypoints from top part

    # right hand keypoints
    ax[0].scatter(
        joints2d_r_i[valid_idx_r, 0],
        joints2d_r_i[valid_idx_r, 1],
        color="r",
        marker="x",
    )
    ax[0].scatter(
        joints2d_l_i[valid_idx_l, 0],
        joints2d_l_i[valid_idx_l, 1],
        color="b",
        marker="x",
    )
    ax[0].set_title(f"{flag} 2D keypoints")

    # GT 2d keypoints (good overlap as it is from perspective camera)
    ax[1].imshow(images_i)
    if not only_hands:
        vis_utils.plot_2d_bbox(bbox2d_b_i, None, "r", ax[1])
        vis_utils.plot_2d_bbox(bbox2d_t_i, None, "b", ax[1])
    ax[1].set_title(f"{flag} 2D bbox")

    # GT 3D keypoints projected to 2D using weak perspective projection
    # (sometimes not completely overlap because of a weak perspective camera)
    ax[2].imshow(images_i)
    if not only_hands:
        ax[2].scatter(kp2d_proj_b_i[:, 0], kp2d_proj_b_i[:, 1], color="r")
        ax[2].scatter(kp2d_proj_t_i[:, 0], kp2d_proj_t_i[:, 1], color="b")
    ax[2].scatter(
        joints2d_proj_r_i[valid_idx_r, 0],
        joints2d_proj_r_i[valid_idx_r, 1],
        color="r",
        marker="x",
    )
    ax[2].scatter(
        joints2d_proj_l_i[valid_idx_l, 0],
        joints2d_proj_l_i[valid_idx_l, 1],
        color="b",
        marker="x",
    )
    ax[2].set_title(f"{flag} 3D keypoints reprojection from cam")

    # GT 3D bbox projected to 2D using weak perspective projection
    # (sometimes not completely overlap because of a weak perspective camera)
    ax[3].imshow(images_i)
    if not only_hands:
        vis_utils.plot_2d_bbox(bbox2d_proj_b_i, None, "r", ax[3])
        vis_utils.plot_2d_bbox(bbox2d_proj_t_i, None, "b", ax[3])
    ax[3].set_title(f"{flag} 3D keypoints reprojection from cam")

    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    fig.tight_layout()
    plt.close()

    im = vis_utils.fig2img(fig)
    return im


def visualize_kps(vis_dict, flag, max_examples, only_hands):
    # visualize keypoints for predition or GT

    images = (vis_dict["vis.images"].permute(0, 2, 3, 1) * 255).numpy().astype(np.uint8)
    K = vis_dict["meta_info.intrinsics"]
    if not only_hands:
        kp2d_b = vis_dict[f"{flag}.object.kp2d.b"].numpy()
        kp2d_t = vis_dict[f"{flag}.object.kp2d.t"].numpy()
        bbox2d_b = vis_dict[f"{flag}.object.bbox2d.b"].numpy()
        bbox2d_t = vis_dict[f"{flag}.object.bbox2d.t"].numpy()

    joints2d_r = vis_dict[f"{flag}.mano.j2d.r"].numpy()
    joints2d_l = vis_dict[f"{flag}.mano.j2d.l"].numpy()

    if not only_hands:
        kp3d_o = vis_dict[f"{flag}.object.kp3d.cam"]
        bbox3d_o = vis_dict[f"{flag}.object.bbox3d.cam"]
        kp2d_proj = tf.project2d_batch(K, kp3d_o)
        kp2d_proj_t, kp2d_proj_b = torch.split(kp2d_proj, [16, 16], dim=1)
        kp2d_proj_t = kp2d_proj_t.numpy()
        kp2d_proj_b = kp2d_proj_b.numpy()

        bbox2d_proj = tf.project2d_batch(K, bbox3d_o)
        bbox2d_proj_t, bbox2d_proj_b = torch.split(bbox2d_proj, [8, 8], dim=1)
        bbox2d_proj_t = bbox2d_proj_t.numpy()
        bbox2d_proj_b = bbox2d_proj_b.numpy()

    # project 3D to 2D using weak perspective camera (not completely overlap)
    joints3d_r = vis_dict[f"{flag}.mano.j3d.cam.r"]
    joints2d_proj_r = tf.project2d_batch(K, joints3d_r).numpy()
    joints3d_l = vis_dict[f"{flag}.mano.j3d.cam.l"]
    joints2d_proj_l = tf.project2d_batch(K, joints3d_l).numpy()

    joints_valid_r = vis_dict["targets.joints_valid_r"]
    joints_valid_l = vis_dict["targets.joints_valid_l"]

    im_list = []
    for idx in range(min(images.shape[0], max_examples)):
        image_id = vis_dict["vis.image_ids"][idx]
        if not only_hands:
            im = visualize_one_example(
                images[idx],
                kp2d_proj_b[idx],
                kp2d_proj_t[idx],
                joints2d_r[idx],
                joints2d_l[idx],
                kp2d_b[idx],
                kp2d_t[idx],
                bbox2d_b[idx],
                bbox2d_t[idx],
                joints2d_proj_r[idx],
                joints2d_proj_l[idx],
                bbox2d_proj_b[idx],
                bbox2d_proj_t[idx],
                joints_valid_r[idx],
                joints_valid_l[idx],
                flag,
                only_hands,
            )
        else:
            im = visualize_one_example(
                images[idx],
                None,
                None,
                joints2d_r[idx],
                joints2d_l[idx],
                None,
                None,
                None,
                None,
                joints2d_proj_r[idx],
                joints2d_proj_l[idx],
                None,
                None,
                joints_valid_r[idx],
                joints_valid_l[idx],
                flag,
                only_hands,
            )
        im_list.append({"fig_name": f"{image_id}__kps", "im": im})
    return im_list


def visualize_rend(
    renderer,
    vertices_r,
    vertices_l,
    mano_faces_r,
    mano_faces_l,
    vertices_o,
    faces_o,
    r_valid,
    l_valid,
    K,
    img,
    only_hands,
    only_obj=False,
):
    # render 3d meshes
    mesh_r = Mesh(v=vertices_r, f=mano_faces_r)
    mesh_l = Mesh(v=vertices_l, f=mano_faces_l)
    if not only_hands:
        mesh_o = Mesh(v=thing.thing2np(vertices_o), f=thing.thing2np(faces_o))

    # render only valid meshes
    meshes = []
    mesh_names = []
    if r_valid:
        meshes.append(mesh_r)
        mesh_names.append("right")

    if l_valid:
        meshes.append(mesh_l)
        mesh_names.append("left")
    if not only_hands:
        meshes = meshes + [mesh_o]
        mesh_names = mesh_names + ["object"]

    if only_obj:
        meshes = [mesh_o]
        mesh_names = ["object"]

    materials = [color2material(mesh_color_dict[name]) for name in mesh_names]

    if len(meshes) > 0:
        # render in image space
        render_img_img = renderer.render_meshes_pose(
            cam_transl=None,
            meshes=meshes,
            image=img,
            materials=materials,
            sideview_angle=None,
            K=K,
        )
        render_img_list = [render_img_img]

        # render rotated meshes
        for angle in list(np.linspace(45, 300, 3)):
            render_img_angle = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=None,
                materials=materials,
                sideview_angle=angle,
                K=K,
            )
            render_img_list.append(render_img_angle)

        # cat all images
        render_img = np.concatenate(render_img_list, axis=0)
        return render_img

    else:
        # dummy image
        render_img = np.concatenate([img] * 4, axis=0)
        return render_img


def visualize_rends(renderer, vis_dict, max_examples, only_hands, return_pil=False):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.right_valid"].bool()
    left_valid = vis_dict["targets.left_valid"].bool()
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_cam = vis_dict["targets.mano.v3d.cam.r"]
    gt_vertices_l_cam = vis_dict["targets.mano.v3d.cam.l"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.mano.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.mano.v3d.cam.l"]

    if not only_hands:
        # object
        gt_obj_v_cam = unpad_vtensor(
            vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
        )  # meter
        pred_obj_v_cam = unpad_vtensor(
            vis_dict["pred.object.v.cam"], vis_dict["pred.object.v_len"]
        )
        pred_obj_f = unpad_vtensor(vis_dict["pred.object.f"], vis_dict["pred.object.f_len"])
    K = vis_dict["meta_info.intrinsics"]

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        image_list.append(images[idx])
        if not only_hands:
            image_gt = visualize_rend(
                renderer,
                gt_vertices_r_cam[idx],
                gt_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                gt_obj_v_cam[idx],
                pred_obj_f[idx],
                r_valid,
                l_valid,
                K_i,
                images[idx],
                only_hands,
            )
        else:
            image_gt = visualize_rend(
                renderer,
                gt_vertices_r_cam[idx],
                gt_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                r_valid,
                l_valid,
                K_i,
                images[idx],
                only_hands,
            )
        image_list.append(image_gt)

        # render pred
        if not only_hands:
            image_pred = visualize_rend(
                renderer,
                pred_vertices_r_cam[idx],
                pred_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                pred_obj_v_cam[idx],
                pred_obj_f[idx],
                r_valid,
                l_valid,
                K_i,
                images[idx],
                only_hands,
            )
        else:
            image_pred = visualize_rend(
                renderer,
                pred_vertices_r_cam[idx],
                pred_vertices_l_cam[idx],
                mano_faces_r,
                mano_faces_l,
                None,
                None,
                r_valid,
                l_valid,
                K_i,
                images[idx],
                only_hands,
            )
        image_list.append(image_pred)

        if not return_pil:
            # stack images into one
            image_pred = vis_utils.im_list_to_plt(
                image_list,
                figsize=(15, 8),
                title_list=["input image", "GT", "pred w/ pred_cam_t"],
            )
        im_list.append(
            {
                "fig_name": f"{image_id}__rend_rvalid={r_valid}, lvalid={l_valid} ",
                "im": image_pred,
            }
        )
    return im_list


def visualize_all(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=False):
    # unpack
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res:
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    # render 3D meshes
    im_list = visualize_rends(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)

    if not return_pil:
        # visualize keypoints
        im_list_kp_gt = visualize_kps(vis_dict, "targets", max_examples, only_hands)
        im_list_kp_pred = visualize_kps(vis_dict, "pred", max_examples, only_hands)

        # concat side by side pred and gt
        for im_gt, im_pred in zip(im_list_kp_gt, im_list_kp_pred):
            im = {
                "fig_name": im_gt["fig_name"],
                "im": vis_utils.concat_pil_images([im_gt["im"], im_pred["im"]]),
            }
            im_list.append(im)

    # post fix image list
    im_list_postfix = []
    for im in im_list:
        im["fig_name"] += postfix
        im_list_postfix.append(im)

    return im_list


def visualize_obj(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=False):
    # unpack
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res:
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    # render 3D meshes
    im_list = visualize_rends_obj(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil)

    if not return_pil:
        # visualize keypoints
        im_list_kp_gt = visualize_kps(vis_dict, "targets", max_examples, only_hands)
        im_list_kp_pred = visualize_kps(vis_dict, "pred", max_examples, only_hands)

        # concat side by side pred and gt
        for im_gt, im_pred in zip(im_list_kp_gt, im_list_kp_pred):
            im = {
                "fig_name": im_gt["fig_name"],
                "im": vis_utils.concat_pil_images([im_gt["im"], im_pred["im"]]),
            }
            im_list.append(im)

    # post fix image list
    im_list_postfix = []
    for im in im_list:
        im["fig_name"] += postfix
        im_list_postfix.append(im)

    return im_list

def visualize_rends_obj(renderer, vis_dict, max_examples, only_hands, return_pil):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.right_valid"].bool()
    left_valid = vis_dict["targets.left_valid"].bool()
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_cam = vis_dict["targets.mano.v3d.cam.r"]
    gt_vertices_l_cam = vis_dict["targets.mano.v3d.cam.l"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.mano.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.mano.v3d.cam.l"]

    # object
    gt_obj_v_cam = unpad_vtensor(
        vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
    )  # meter
    pred_obj_v_cam = unpad_vtensor(
        vis_dict["pred.object.v.cam"], vis_dict["pred.object.v_len"]
    )
    pred_obj_f = unpad_vtensor(vis_dict["pred.object.f"], vis_dict["pred.object.f_len"])
    K = vis_dict["meta_info.intrinsics"]

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        # render gt
        image_list = []
        image_list.append(images[idx])
        # image_gt = visualize_rend(
        #     renderer,
        #     gt_vertices_r_cam[idx],
        #     gt_vertices_l_cam[idx],
        #     mano_faces_r,
        #     mano_faces_l,
        #     gt_obj_v_cam[idx],
        #     pred_obj_f[idx],
        #     r_valid,
        #     l_valid,
        #     K_i,
        #     images[idx],
        #     only_hands,
        #     only_obj=True,
        # )
        # image_list.append(image_gt)

        # render pred
        image_pred = visualize_rend(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            mano_faces_r,
            mano_faces_l,
            pred_obj_v_cam[idx],
            pred_obj_f[idx],
            r_valid,
            l_valid,
            K_i,
            images[idx],
            only_hands,
            only_obj=True,
        )
        image_list.append(image_pred)

        if not return_pil:
            # stack images into one
            image_pred = vis_utils.im_list_to_plt(
                image_list,
                figsize=(15, 8),
                title_list=["input image", "GT", "pred w/ pred_cam_t"],
            )
        im_list.append(
            {
                "fig_name": f"{image_id}__rend_rvalid={r_valid}, lvalid={l_valid} ",
                "im": image_pred,
            }
        )
    return im_list


def visualize_gif(vis_dict, max_examples, renderer, postfix, no_tqdm, only_hands=False, return_pil=False, **kwargs):
    # unpack
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    if images.shape[-1] != renderer.img_res:
        images = F.interpolate(images, size=renderer.img_res, mode="bilinear", align_corners=True)

    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    # render 3D meshes
    # im_list = visualize_rends(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil) # for other visualizations
    gif_list = visualize_rends_gifs(renderer, vis_dict, max_examples, only_hands, return_pil=return_pil, **kwargs) # for gifs

    # post fix image list
    # im_list_postfix = []
    # for im in im_list:
    #     im["fig_name"] += postfix
    #     im_list_postfix.append(im)

    return [], gif_list

def visualize_rends_gifs(renderer, vis_dict, max_examples, only_hands, return_pil=False, **kwargs):
    # render meshes

    # unpack data
    image_ids = vis_dict["vis.image_ids"]
    right_valid = vis_dict["targets.right_valid"].bool()
    left_valid = vis_dict["targets.left_valid"].bool()
    images = vis_dict["vis.images"].permute(0, 2, 3, 1).numpy()
    gt_vertices_r_cam = vis_dict["targets.mano.v3d.cam.r"]
    gt_vertices_l_cam = vis_dict["targets.mano.v3d.cam.l"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]
    pred_vertices_r_cam = vis_dict["pred.mano.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["pred.mano.v3d.cam.l"]

    crop_viz = kwargs.get('crop_viz', False)
    if crop_viz:
        # bbox_r = vis_dict["inputs.r_bbox"] # this is need for rendering hamer predictions
        # bbox_l = vis_dict["inputs.l_bbox"]
        crop_img_r = vis_dict['crop_r'].numpy().astype(np.uint8)
        crop_img_l = vis_dict['crop_l'].numpy().astype(np.uint8)
        mano_faces_l = mano_faces_r # render using right mano and flip

    if not only_hands:
        # object
        gt_obj_v_cam = unpad_vtensor(
            vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
        )  # meter
        pred_obj_v_cam = unpad_vtensor(
            vis_dict["pred.object.v.cam"], vis_dict["pred.object.v_len"]
        )
        pred_obj_f = unpad_vtensor(vis_dict["pred.object.f"], vis_dict["pred.object.f_len"])
    K = vis_dict["meta_info.intrinsics"]

    # rendering
    im_list = []
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        if crop_viz:
            crop_r = Image.fromarray(crop_img_r[idx])
            crop_l = Image.fromarray(crop_img_l[idx])
            crop_r = crop_r.resize((renderer.img_res, renderer.img_res))
            crop_l = crop_l.resize((renderer.img_res, renderer.img_res))
            crop_r = np.array(crop_r) / 255
            crop_l = np.array(crop_l) / 255
            kwargs['crop_r'] = crop_r
            kwargs['crop_l'] = crop_l

        # render gt
        image_list = []
        image_list.append(images[idx])
        image_hand_r = visualize_rend_gif(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            mano_faces_r,
            mano_faces_l,
            None,
            None,
            r_valid,
            l_valid,
            K_i,
            images[idx],
            only_hands,
            only_right=True,
            **kwargs,
        )

        image_hand_l = visualize_rend_gif(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            mano_faces_r,
            mano_faces_l,
            None,
            None,
            r_valid,
            l_valid,
            K_i,
            images[idx],
            only_hands,
            only_left=True,
            **kwargs,
        )

        im_list.append(
            {
                "fig_name": f"{image_id}__rend_rvalid={r_valid}, lvalid={l_valid} ",
                "im_right": image_hand_r,
                "im_left": image_hand_l,
            }
        )
    return im_list

def scale_box(bbox, scale=1.2, res=224):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w >= res-4 or h >= res-4:
        return bbox

    x1 -= w*(scale-1)/2
    y1 -= h*(scale-1)/2
    x2 += w*(scale-1)/2
    y2 += h*(scale-1)/2
    # clip the values in [0,res]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(res, x2)
    y2 = min(res, y2)
    return [x1, y1, x2, y2]

def visualize_rend_gif(
    renderer,
    vertices_r,
    vertices_l,
    mano_faces_r,
    mano_faces_l,
    vertices_o,
    faces_o,
    r_valid,
    l_valid,
    K,
    img,
    only_hands,
    only_obj=False,
    only_right=False,
    only_left=False,
    **kwargs,
):
    # render 3d meshes
    mesh_r = Mesh(v=vertices_r, f=mano_faces_r)
    mesh_l = Mesh(v=vertices_l, f=mano_faces_l)
    if not only_hands:
        mesh_o = Mesh(v=thing.thing2np(vertices_o), f=thing.thing2np(faces_o))

    # render only valid meshes
    meshes = []
    mesh_names = []
    if r_valid and not only_left:
        meshes.append(mesh_r)
        mesh_names.append("right")

    if l_valid and not only_right:
        meshes.append(mesh_l)
        mesh_names.append("left")
    if not only_hands:
        meshes = meshes + [mesh_o]
        mesh_names = mesh_names + ["object"]

    if only_obj:
        meshes = [mesh_o]
        mesh_names = ["object"]

    materials = [color2material(mesh_color_dict[name], metallicFactor=0.5) for name in mesh_names]

    if len(meshes) > 0:
        # render in image space
        if kwargs.get('crop_viz', False):
            render_img_img = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=kwargs['crop_r'] if only_right else kwargs['crop_l'],
                materials=materials,
                sideview_angle=None,
                K=K,
            )
        else:
            render_img_img = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=img,
                materials=materials,
                sideview_angle=None,
                K=K,
            )
        render_img_list = [render_img_img]

        cam_transl = np.zeros((3,))
        intrx = K.clone()
        assert len(meshes) == 1 # only one hand at a time
        verts = meshes[0].v.copy()
        cam_transl[0] = -verts[:,0].mean() # different convention?
        cam_transl[1] = -verts[:,1].mean() # different convention?
        z = verts[:,2].mean()
        scale_factor = z / 0.2 # rendering depth
        if kwargs.get('scale_factor', None) is not None:
            scale_factor = kwargs['scale_factor']
        intrx[0,0] *= scale_factor
        intrx[1,1] *= scale_factor

        # render rotated meshes
        for angle in list(np.linspace(0, 360, 21)):
            render_img_angle = renderer.render_meshes_pose(
                cam_transl=cam_transl,
                meshes=meshes,
                image=None,
                materials=materials,
                sideview_angle=angle,
                K=intrx,
            )
            render_img_list.append(render_img_angle)

        # render only hand in the camera frame
        render_img_hand = renderer.render_meshes_pose(
                cam_transl=None,
                meshes=meshes,
                image=None,
                materials=materials,
                sideview_angle=None,
                K=K,
            )
        render_img_list.append(render_img_hand)

        # cat all images
        render_img = np.concatenate(render_img_list, axis=0)
        return render_img

    else:
        # dummy image
        render_img = np.concatenate([img] * 22, axis=0) # 21 rotations + 1 hand only in camera frame
        return render_img