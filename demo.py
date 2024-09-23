from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import pickle

from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full

from wildhands.configs.parser import construct_args
from wildhands.models.wrapper import WildHandsWrapper as Wrapper
from wildhands.datasets.dataset import WildHandsDataset
import wildhands.common.data_utils as data_utils

from vitpose_model import ViTPoseModel

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--hamer_ckpt', type=str, default=None, help='Path to pretrained model checkpoint')
    parser.add_argument('--wildhands_ckpt', type=str, default=None, help='Path to pretrained WildHands model checkpoint')
    parser.add_argument('--img_folder', type=str, default='downloads/example_data', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--render_res', type=int, default=840, help='Resolution for rendering')
    parser.add_argument('--focal_length', type=float, default=1000, help='Camera focal length corresponding to the input image')
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if args.hamer_ckpt is not None: # Setup HaMeR model
        assert args.wildhands_ckpt is None, 'Cannot use both HaMeR and WildHands models together'
        model, cfg = load_hamer(args.hamer_ckpt)
        model = model.to(device)
        model.eval()

        renderer = Renderer(cfg, faces=model.mano.faces)

    elif args.wildhands_ckpt is not None: # Setup WildHands model
        cfg = construct_args()
        model = Wrapper(cfg)
        ckpt = torch.load(args.wildhands_ckpt, map_location='cpu')
        ckpt_params = {}
        redundant_keys = ['head_o', 'arti_head', 'grasp_classifier']
        for k, v in ckpt["state_dict"].items():
            if not any([rk in k for rk in redundant_keys]):
                ckpt_params[k] = v
        model.load_state_dict(ckpt_params)
        model = model.to(device)
        model.eval()

        renderer = Renderer(cfg, faces = model.model.mano_r.mano.faces)
    
    else:
        raise ValueError('Please provide either HaMeR or WildHands checkpoint')

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])

    # Iterate over all images in folder
    for img_path in img_paths:

        # square images are convenient since different models have different input sizes
        cv_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        input_res = max(cv_img.shape[:2])
        image = data_utils.generate_patch_image_clean(cv_img, [cv_img.shape[1]/2, cv_img.shape[0]/2, input_res, input_res], 1.0, 0.0, [args.render_res, args.render_res], cv2.INTER_CUBIC)[0]
        img = image.clip(0, 255)
        img_cv2 = img.astype(np.uint8)[..., ::-1]

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        scaled_focal_length = args.focal_length * args.render_res / input_res
        if args.hamer_ckpt is not None:
            dataset = ViTDetDataset(cfg, img_cv2, boxes, right, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        elif args.wildhands_ckpt is not None:
            dataset = WildHandsDataset(cfg, img, boxes, right, focal_length=scaled_focal_length, rescale_factor=1.75)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            if args.hamer_ckpt is not None: # HaMeR predictions
                batch_right = batch['right']
                multiplier = (2 * batch_right - 1)
                pred_cam = out['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            elif args.wildhands_ckpt is not None: # WildHands predictions
                batch_right = batch[1]['right'] # batch = (inputs, meta_info)
                pred_cam_t_full_r_wh = out['pred.cam_t.r'].cpu().numpy()
                pred_cam_t_full_l_wh = out['pred.cam_t.l'].cpu().numpy()
                pred_vertices_r = out['pred.vertices.r'].cpu().numpy()
                pred_vertices_l = out['pred.vertices.l'].cpu().numpy()
            
            batch_size = batch_right.shape[0]
            for n in range(batch_size):

                # Add all verts and cams to list
                is_right = batch_right[n].cpu().numpy()
                if args.hamer_ckpt is not None:
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    verts[:,0] = (2*is_right-1)*verts[:,0]
                    cam_t = pred_cam_t_full[n]
                elif args.wildhands_ckpt is not None:
                    verts = pred_vertices_r[n] if is_right else pred_vertices_l[n]
                    cam_t = pred_cam_t_full_r_wh[n] if is_right else pred_cam_t_full_l_wh[n]
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

        # Render hands onto the image
        if len(all_verts) > 0:
            misc_args = dict(mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1), focal_length=scaled_focal_length)

            input_img = cv2.resize(img_cv2, (args.render_res, args.render_res), interpolation=cv2.INTER_CUBIC)
            input_img = input_img.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)

            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=[args.render_res, args.render_res], is_right=all_right, **misc_args)
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            
            # Get filename from path img_path
            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])

if __name__ == '__main__':
    main()
