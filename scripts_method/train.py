import os
import os.path as op
import sys
from pprint import pformat

import comet_ml
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

sys.path.append(".")

import common.comet_utils as comet_utils
import common.tb_utils as tb_utils
import src.factory as factory
from common.torch_utils import reset_all_seeds
from src.utils.const import args

# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('medium')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(args):
    if args.experiment is not None:
        comet_utils.log_exp_meta(args)
    else:
        tb_utils.log_exp_meta(args)
    reset_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = factory.fetch_model(args)
    
    if args.load_ckpt != "":
        ckpt = torch.load(args.load_ckpt, map_location='cpu')
        wrapper.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info(f"Loaded weights from {args.load_ckpt}")
    
    ckpt_callback = ModelCheckpoint(
        monitor="loss__val",
        verbose=True,
        save_top_k=3,
        mode="min",
        every_n_epochs=args.eval_every_epoch,
        save_last=True,
        dirpath=op.join(args.log_dir, "checkpoints"),
    )

    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)

    if args.logger == "comet":
        pl_logger = None
    elif args.logger == "tensorboard":
        pl_logger = pl.loggers.TensorBoardLogger(args.log_dir)
    model_summary_cb = ModelSummary(max_depth=3)
    callbacks = [ckpt_callback, pbar_cb, model_summary_cb]
    trainer = pl.Trainer(
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.acc_grad,
        devices=-1,
        accelerator="gpu",
        logger=pl_logger,
        min_epochs=args.num_epoch,
        max_epochs=args.num_epoch,
        callbacks=callbacks,
        log_every_n_steps=args.log_every,
        default_root_dir=args.log_dir,
        check_val_every_n_epoch=args.eval_every_epoch,
        num_sanity_val_steps=1,
        enable_model_summary=False,
        strategy='ddp_find_unused_parameters_true',
    )

    reset_all_seeds(args.seed)
    train_loader = factory.fetch_dataloader(args, "train")
    logger.info(f"Hyperparameters: \n {pformat(args)}")
    logger.info("*** Started training ***")
    reset_all_seeds(args.seed)
    ckpt_path = None if args.ckpt_p == "" else args.ckpt_p
    val_loaders = [factory.fetch_dataloader(args, "val")]
    wrapper.set_training_flags()
    trainer.fit(wrapper, train_loader, val_loaders, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main(args)
