import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from common.torch_utils import reset_all_seeds
from common.tb_utils import push_images
from src.datasets.assembly_dataset import AssemblyDataset
from src.datasets.hands_light_dataset import HandsLightDataset
from src.datasets.epic_dataset import EPICDataset
from src.datasets.epic_grasp_dataset import EPICGraspDataset
from src.datasets.epic_seg_dataset import EPICSegDataset
from src.datasets.epic_depth_dataset import EPICDepthDataset
from src.datasets.ego_grasp_dataset import Ego4DGraspDataset
from src.datasets.ego_seg_dataset import Ego4DSegDataset
from src.datasets.h2o_dataset import H2ODataset
from src.datasets.ego_exo_dataset import EgoExoDataset


def fetch_dataset_devel(args, is_train, seq=None):
    split = args.trainsplit if is_train else args.valsplit
    if args.method in ["hands_light", "hamer_light", "handoccnet_light", "arctic_light"]:
        DATASET = HandsLightDataset
    else:
        assert False
    if seq is not None:
        split = args.run_on

    relevant_dataset = None
    if is_train:
        relevant_dataset = args.get('dataset', None)
    else:
        relevant_dataset = args.get('val_dataset', None)

    if relevant_dataset is None:
        ds = DATASET(args=args, split=split, seq=seq)
    else:
        sets = relevant_dataset.split('+')
        ds = []
        for s in sets:
            if s == 'assembly':
                curr_ds = AssemblyDataset(args, split)
                ds.append(curr_ds)
            elif s == 'hands':
                curr_ds = HandsLightDataset(args, split)
                ds.append(curr_ds)
            elif s == 'epic':
                curr_ds = EPICDataset(args, split)
                ds.append(curr_ds)
            elif s == 'epic_grasp':
                curr_ds = EPICGraspDataset(args, split)
                ds.append(curr_ds)
            elif s == 'epic_seg':
                curr_ds = EPICSegDataset(args, split)
                ds.append(curr_ds)
            elif s == 'epic_depth':
                curr_ds = EPICDepthDataset(args, split)
                ds.append(curr_ds)
            elif s == 'ego_grasp':
                curr_ds = Ego4DGraspDataset(args, split)
                ds.append(curr_ds)
            elif s == 'ego_seg':
                curr_ds = Ego4DSegDataset(args, split)
                ds.append(curr_ds)
            elif s == 'h2o':
                curr_ds = H2ODataset(args, split)
                ds.append(curr_ds)
            elif s == 'egoexo':
                curr_ds = EgoExoDataset(args, split)
                ds.append(curr_ds)
            else:
                assert False
        
        ds = torch.utils.data.ConcatDataset(ds) # mixed dataset training
    return ds


def collate_custom_fn(data_list):
    data = data_list[0]
    _inputs, _targets, _meta_info = data
    out_inputs = {}
    out_targets = {}
    out_meta_info = {}

    for key in _inputs.keys():
        out_inputs[key] = []

    for key in _targets.keys():
        out_targets[key] = []

    for key in _meta_info.keys():
        out_meta_info[key] = []

    for data in data_list:
        inputs, targets, meta_info = data
        for key, val in inputs.items():
            out_inputs[key].append(val)

        for key, val in targets.items():
            out_targets[key].append(val)

        for key, val in meta_info.items():
            out_meta_info[key].append(val)

    for key in _inputs.keys():
        out_inputs[key] = torch.cat(out_inputs[key], dim=0)

    for key in _targets.keys():
        out_targets[key] = torch.cat(out_targets[key], dim=0)

    for key in _meta_info.keys():
        if key not in ["imgname", "query_names"]:
            out_meta_info[key] = torch.cat(out_meta_info[key], dim=0)
        else:
            out_meta_info[key] = sum(out_meta_info[key], [])

    return out_inputs, out_targets, out_meta_info


def fetch_dataloader(args, mode, seq=None):
    if mode == "train":
        reset_all_seeds(args.seed)
        dataset = fetch_dataset_devel(args, is_train=True)
        if type(dataset) == HandsLightDataset or type(dataset) == AssemblyDataset or type(dataset) == EPICDataset or type(dataset) == EPICDepthDataset \
            or type(dataset) == EPICGraspDataset or type(dataset) == EPICSegDataset or type(dataset) == Ego4DGraspDataset or type(dataset) == Ego4DSegDataset \
            or type(dataset) == H2ODataset or type(dataset) == EgoExoDataset or type(dataset) == ConcatDataset:
            collate_fn = None
        else:
            collate_fn = collate_custom_fn
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=args.shuffle_train,
            collate_fn=collate_fn,
        )

    elif mode == "val" or mode == "eval":
        if "submit_" in args.extraction_mode: # this is not used in the current framework
            raise NotImplementedError
        else:
            dataset = fetch_dataset_devel(args, is_train=False, seq=seq)
        if type(dataset) in [HandsLightDataset, AssemblyDataset, ConcatDataset, EPICDataset, EPICGraspDataset, \
            EPICDepthDataset, EPICSegDataset, Ego4DGraspDataset, Ego4DSegDataset, H2ODataset, EgoExoDataset]:
            collate_fn = None
        else:
            collate_fn = collate_custom_fn
        return DataLoader(
            dataset=dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    else:
        assert False


def fetch_model(args):
    if args.method in ["arctic_light"]:
        from src.models.arctic_light.wrapper import ArcticSFWrapper as Wrapper
    elif args.method in ["hamer_light"]:
        from src.models.hamer_light.wrapper import HaMeRWrapper as Wrapper
    elif args.method in ["handoccnet_light"]:
        from src.models.handoccnet_light.wrapper import HandOccNetWrapper as Wrapper
    elif args.method in ["hands_light"]:
        from src.models.hands_light.wrapper import HandsWrapper as Wrapper
    else:
        assert False, f"Invalid method ({args.method})"
    
    if args.logger == "comet":
        model = Wrapper(args)
    elif args.logger == "tensorboard":
        model = Wrapper(args, push_images_fn=push_images)
    return model
