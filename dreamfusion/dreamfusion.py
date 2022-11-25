import gc
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.utils.checkpoint
from PIL import features
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, whoami
from six import StringIO

from extensions.stable_dreamfusion_extension.dreamfusion import conversion
from extensions.stable_dreamfusion_extension.dreamfusion.df_config import DreamfusionConfig
#from extensions.stable_dreamfusion_extension.dreamfusion.xattention import save_pretrained
from modules import paths, shared, devices, sd_models

try:
    cmd_dreamfusion_models_path = shared.cmd_opts.dreamfusion_models_path
except:
    cmd_dreamfusion_models_path = None


def get_df_models():
    model_dir = os.path.dirname(cmd_dreamfusion_models_path) if cmd_dreamfusion_models_path else paths.models_path
    out_dir = os.path.join(model_dir, "dreamfusion")
    output = []
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isdir(os.path.join(out_dir, found)):
                output.append(found)
    return output


def load_params(model_dir):
    data = DreamfusionConfig().from_file(model_dir)

    target_values = ["text",
                     "negative",
                     "o",
                     "o2",
                     "test",
                     "save_mesh",
                     "eval_interval",
                     "workspace",
                     "guidance",
                     "seed",
                     "iters",
                     "lr",
                     "ckpt",
                     "cuda_ray",
                     "max_steps",
                     "num_steps",
                     "upsample_steps",
                     "update_extra_interval",
                     "max_ray_batch",
                     "albedo",
                     "albedo_iters",
                     "uniform_sphere_rate",
                     "bg_radius",
                     "density_thresh",
                     "fp16",
                     "backbone",
                     "w",
                     "h",
                     "jitter_pose",
                     "bound",
                     "dt_gamma",
                     "min_near",
                     "radius_range",
                     "fovy_range",
                     "dir_text",
                     "suppress_face",
                     "angle_overhead",
                     "angle_front",
                     "lambda_entropy",
                     "lambda_opacity",
                     "orient",
                     "lambda_smooth",
                     "gui",
                     "gui_w",
                     "gui_h",
                     "gui_radius",
                     "gui_fovy",
                     "gui_light_theta",
                     "gui_light_phi",
                     "max_spp"
                     ]

    values = []
    for target in target_values:
        if target in data:
            value = data[target]
            if target == "max_token_length":
                value = str(value)
            values.append(value)
        else:
            values.append(None)
    values.append(f"Loaded params from {model_dir}.")
    return values


def start_training(text,
                   negative,
                   o,
                   o2,
                   test,
                   save_mesh,
                   eval_interval,
                   workspace,
                   guidance,
                   seed,
                   iters,
                   lr,
                   ckpt,
                   cuda_ray,
                   max_steps,
                   num_steps,
                   upsample_steps,
                   update_extra_interval,
                   max_ray_batch,
                   albedo,
                   albedo_iters,
                   uniform_sphere_rate,
                   bg_radius,
                   density_thresh,
                   fp16,
                   backbone,
                   w,
                   h,
                   jitter_pose,
                   bound,
                   dt_gamma,
                   min_near,
                   radius_range,
                   fovy_range,
                   dir_text,
                   suppress_face,
                   angle_overhead,
                   angle_front,
                   lambda_entropy,
                   lambda_opacity,
                   orient,
                   lambda_smooth,
                   gui,
                   gui_w,
                   gui_h,
                   gui_radius,
                   gui_fovy,
                   gui_light_theta,
                   gui_light_phi,
                   max_spp
                   ):
    return "", ""