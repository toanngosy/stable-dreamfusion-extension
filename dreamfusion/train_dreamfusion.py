# From shivam shiaro's repo, with modification to work with stable-dreamfusion
import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import random
import traceback
import numpy as np

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler, \
    StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import logging as dl
from huggingface_hub import HfFolder, whoami
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from extensions.stable_dreamfusion_extension.dreamfusion import xattention
from extensions.stable_dreamfusion_extension.dreamfusion.dreamfusion import dumb_safety, save_checkpoint, list_features, \
    is_image, printm

from modules import shared
from stable_dreamfusion.nerf.provider import NeRFDataset
from stable_dreamfusion.nerf.utils import *
from stable_dreamfusion.nerf.optimizer import Shampoo




# Custom stuff
try:
    cmd_dreamfusion_models_path = shared.cmd_opts.dreamfusion_models_path
except:
    cmd_dreamfusion_models_path = None
    
mem_record = {}
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
dl.set_verbosity_error()

def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    except:
        pass
    printm("Cleanup completed.")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=512, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo', action='store_true', help="only use albedo shading to train, overrides --albedo_iters")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='grid', help="nerf backbone, choose from [grid, vanilla]")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true', help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    parser.add_argument('--lambda_entropy', type=float, default=1e-4, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=0, help="loss scale for surface smoothness")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    # TODO: distributed training
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    if args.o:
        args.fp16 = True
        args.dir_text = True
        args.cuda_ray =True
    elif args.o2:
        if args.albedo:
            args.fp16 = True
        args.dir_text = True
        args.backbone = 'vanilla'
    
    if args.albedo:
        args.albedo_iters = args.iters
    
    if args.backbone == 'vanilla':
        from stable_dreamfusion.nerf.network import NERFNetwork
    elif args.backbone == 'grid':
        from stable_dreamfusion.nerf.network import NERFNetwork
    else:
        raise NotImplementedError(f'--backbone {args.backbone} is not implemented')
    return args

def main(args, memory_record):
    args.tokenizer_name = None
    global mem_record
    mem_record = memory_record
    logging_dir = Path(args.output_dir, "logging")
    args.max_token_length = int(args.max_token_length)
    if not args.pad_tokens and args.max_token_length > 75:
        logger.debug("Cannot raise token length limit above 75 when pad_tokens=False")

    # if args.attention == "xformers":
    #     xattention.replace_unet_cross_attn_to_xformers()
    # elif args.attention == "flash_attention":
    #     xattention.replace_unet_cross_attn_to_flash_attention()
    # else:
    #     xattention.replace_unet_cross_attn_to_default()

    # weight_dtype = torch.float32
    # if args.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif args.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    
    model = NERFNetwork(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer('df', args, model, guidance, device=device, workspace=args.workspace, fp16=args.fp16, use_checkpoint=args.ckpt)


        test_loader = NeRFDataset(args, device=device, type='test', H=args.H, W=args.W, size=100).dataloader()
        trainer.test(test_loader)
        
        if args.save_mesh:
            trainer.save_mesh(resolution=256)
            
    
    else:
        
        train_loader = NeRFDataset(args, device=device, type='train', H=args.h, W=args.w, size=100).dataloader()

        optimizer = lambda model: torch.optim.Adam(model.get_params(args.lr), betas=(0.9, 0.99), eps=1e-15)
        # optimizer = lambda model: Shampoo(model.get_params(opt.lr))

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / args.iters, 1))
        # scheduler = lambda optimizer: optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=opt.iters, pct_start=0.1)

        if args.guidance == 'stable-diffusion':
            from stable_dreamfusion.nerf.sd import StableDiffusion
            guidance = StableDiffusion(device)
        elif args.guidance == 'clip':
            from stable_dreamfusion.nerf.clip import CLIP
            guidance = CLIP(device)
        else:
            raise NotImplementedError(f'--guidance {args.guidance} is not implemented.')

        trainer = Trainer('df', args, model, guidance, 
                          device=device, workspace=args.workspace, 
                          optimizer=optimizer, ema_decay=None, 
                          fp16=args.fp16, lr_scheduler=scheduler, 
                          use_checkpoint=args.ckpt, eval_interval=args.eval_interval, 
                          scheduler_update_every_step=True)

        valid_loader = NeRFDataset(args, device=device, type='val', H=args.H, W=args.W, size=5).dataloader()

        max_epoch = np.ceil(args.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

    
    pass