import asyncio
from typing import Optional, List

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

import modules.script_callbacks as script_callbacks
#from extensions.stable_dreamfusion_extension.dreamfusion import dreamfusion
from webui import wrap_gradio_gpu_call

class DreamfusionParameters(BaseModel):
    df_text: str = "a hamburger"
    df_negative: str
    df_o: bool = False
    df_o2: bool = False
    df_test: bool = False
    df_save_mesh: bool = False
    df_eval_interval: int = 10
    df_workspace: str = "workspace"
    df_guidance: str = "stable-diffusion"
    df_seed: int = 0

    # training options
    df_iters: int = 10000
    df_lr: float = 1e-3
    df_ckpt: str = "latest"
    df_cuda_ray: bool = False
    df_max_steps: int = 512
    df_num_steps: int = 64
    df_upsample_steps: int = 32
    df_update_extra_interval: int = 16
    df_max_ray_batch: int = 4096
    df_albedo: bool = False
    df_albedo_iters: int = 1000
    df_uniform_sphere_rate: float = 0.5

    # model options
    df_bg_radius: float = 1.4
    df_density_thresh: float = 10
    
    # network backbone
    df_fp16: bool = False
    df_backbone: str = "grid"

    # rendering resolution in training, decrease this if CUDA OOM
    df_w: int = 64
    df_h: int = 64
    df_jitter_pose: bool = False

    # dataset options
    df_bound: float = 1
    df_dt_gamma: float = 0
    df_min_near: float = 0.1
    df_radius_range: List[float] = [1.0, 1.5]
    df_fovy_range: List[float] = [40, 70]
    df_dir_text: bool = False
    df_suppress_face: bool = False
    df_angle_overhead: float = 30
    df_angle_front: float = 60
    
    df_lambda_entropy: float = 1e-4
    df_lambda_opacity: float = 0
    df_orient: float = 1e-2
    df_lambda_smooth: float: 0
    
    # GUI options
    df_gui: bool = False
    df_gui_w: int = 800
    df_gui_h: int = 800
    df_gui_radius: float = 3
    df_gui_fovy: float = 60
    df_gui_light_theta: float = 60
    df_gui_light_phi: float = 0
    df_max_spp: int = 1
