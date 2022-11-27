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
from extensions.stable_dreamfusion_extension.dreamfusion.xattention import save_pretrained
from modules import paths, shared, devices, sd_models

try:
    cmd_dreamfusion_models_path = shared.cmd_opts.dreamfusion_models_path
except:
    cmd_dreamfusion_models_path = None

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

mem_record = {}

StableDiffusionPipeline.save_pretrained = save_pretrained


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


def printm(msg, reset=False):
    global mem_record
    if not mem_record:
        mem_record = {}
    if reset:
        mem_record = {}
    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
    cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
    mem_record[msg] = f"{allocated}/{cached}GB"
    print(f' {msg} \n Allocated: {allocated}GB \n Reserved: {cached}GB \n')


def load_params(model_dir):
    data = DreamfusionConfig().from_file(model_dir)

    target_values = ["model_dir",
                     "half_model",
                     "text",
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


def start_training(model_dir,
                   half_model,
                   text,
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
    print('here')
    print('model_dir', model_dir) 
    print('half_model', half_model)
    print('text', text)
    print('negative', negative)
    print('o', o)
    print('o2', o2)
    print('test', test)
    print('save_mesh', save_mesh)
    print('eval_interval', eval_interval)
    print('workspace', workspace)
    print('guidance', guidance)
    print('seed', seed)
    print('iters', iters)
    print('lr', lr)
    print('ckpt', ckpt)
    print('cuda_ray', cuda_ray)
    print('max_steps', max_steps)
    print('num_steps', num_steps)
    print('upsample_steps', upsample_steps)
    print('update_extra_interval', update_extra_interval)
    print('max_ray_batch', max_ray_batch)
    print('albedo', albedo)
    print('albedo_iters', albedo_iters)
    print('uniform_sphere_rate', uniform_sphere_rate)
    print('bg_radius', bg_radius)
    print('density_thresh', density_thresh)
    print('fp16', fp16)
    print('backbone', backbone)
    print('w', w)
    print('h', h)
    print('jitter_pose', jitter_pose)
    print('bound', bound)
    print('dt_gamma', dt_gamma)
    print('min_near', min_near)
    print('radius_range', radius_range)
    print('fovy_range', fovy_range)
    print('dir_text', dir_text)
    print('suppress_face', suppress_face)
    print('angle_overhead', angle_overhead)
    print('angle_front', angle_front)
    print('lambda_entropy', lambda_entropy)
    print('lambda_opacity', lambda_opacity)
    print('orient', orient)
    print('lambda_smooth', lambda_smooth)
    print('gui', gui)
    print('gui_w', gui_w)
    print('gui_h', gui_h)
    print('gui_radius', gui_radius)
    print('gui_fovy', gui_fovy)
    print('gui_light_theta', gui_light_theta)
    print('gui_light_phi', gui_light_phi)
    print('max_spp', max_spp)
    global mem_record
    if model_dir == "" or model_dir is None:
        print("Invalid model name.")
        return "Create or select a model first.", ""
    # delete me
    if model_dir != "":
        print("Model name goes through.")
        return "Model here", ""
    
    config = DreamfusionConfig().from_ui(model_dir,
                                         half_model,
                                         text,
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
                                         )

    # Clear pretrained VAE Name if applicable
    if "pretrained_vae_name_or_path" in config.__dict__:
        if config.pretrained_vae_name_or_path == "":
            config.pretrained_vae_name_or_path = None
    else:
        config.pretrained_vae_name_or_path = None

    config.save()
    msg = None

    if msg:
        shared.state.textinfo = msg
        print(msg)
        return msg, msg

    print("Starting Dreamfusion training...")
    if shared.sd_model is not None:
        shared.sd_model.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
    printm("VRAM cleared.", True)
    shared.state.textinfo = "Initializing dreamfusion training..."
    from extensions.stable_dreamfusion_extension.dreamfusion.train_dreamfusion import main
    config, mem_record = main(config, mem_record)
    if config.revision != total_steps:
        config.save()
    total_steps = config.revision
    devices.torch_gc()
    gc.collect()
    printm("Training completed, reloading SD Model.")
    print(f'Memory output: {mem_record}')
    if shared.sd_model is not None:
        shared.sd_model.to(shared.device)
    print('Re-applying optimizations...')
    res = f"Training {'interrupted' if shared.state.interrupted else 'finished'}. " \
          f"Total lifetime steps: {total_steps} \n"
    print(f"Returning result: {res}")
    return res, ""


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


# TODO: use_half?
def save_checkpoint(model_name: str, vae_path: str, total_steps: int, use_half: bool = False):
    print(f"Successfully trained model for a total of {total_steps} steps, converting to ckpt.")
    ckpt_dir = shared.cmd_opts.ckpt_dir
    models_path = os.path.join(paths.models_path, "Stable-diffusion")
    if ckpt_dir is not None:
        models_path = ckpt_dir
    src_path = os.path.join(
        os.path.dirname(cmd_dreamfusion_models_path) if cmd_dreamfusion_models_path else paths.models_path, "dreamfusion",
        model_name, "working")
    out_file = os.path.join(models_path, f"{model_name}_{total_steps}.ckpt")
    conversion.diff_to_sd(src_path, vae_path, out_file, use_half)
    sd_models.list_models()