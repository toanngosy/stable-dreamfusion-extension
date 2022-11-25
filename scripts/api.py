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
    df_lambda_smooth: float = 0
    
    # GUI options
    df_gui: bool = False
    df_gui_w: int = 800
    df_gui_h: int = 800
    df_gui_radius: float = 3
    df_gui_fovy: float = 60
    df_gui_light_theta: float = 60
    df_gui_light_phi: float = 0
    df_max_spp: int = 1


def dreamFusionAPI(demo: gr.Blocks, app: FastAPI):
    @app.post("dreamfusion/create_model")
    async def create_model(
            name,
            source,
            scheduler,
            model_url,
            hub_token):
        print(f"Creating new Checkpoint: {name}")
        fn = conversion.extract_checkpoint(name, source, scheduler, model_url, hub_token)

    @app.post("dreamfusion/start_training")
    async def start_training(params: DreamfusionParameters):
        print("Start Training")
        task = asyncio.create_task(train_model(params))
        return {"status": "finished"}

    async def train_model(params: DreamfusionParameters):
        fn = wrap_gradio_gpu_call(dreamfusion.start_training(
            params.df_text,
            params.df_negative,
            params.df_o,
            params.df_o2,
            params.df_test,
            params.df_save_mesh,
            params.df_eval_interval,
            params.df_workspace,
            params.df_guidance,
            params.df_seed,
            params.df_iters,
            params.df_lr,
            params.df_ckpt,
            params.df_cuda_ray,
            params.df_max_steps,
            params.df_num_steps,
            params.df_upsample_steps,
            params.df_update_extra_interval,
            params.df_max_ray_batch,
            params.df_albedo,
            params.df_albedo_iters,
            params.df_uniform_sphere_rate,
            params.df_bg_radius,
            params.df_density_thresh,
            params.df_fp16,
            params.df_backbone,
            params.df_w,
            params.df_h,
            params.df_jitter_pose,
            params.df_bound,
            params.df_dt_gamma,
            params.df_min_near,
            params.df_radius_range,
            params.df_fovy_range,
            params.df_dir_text,
            params.df_suppress_face,
            params.df_angle_overhead,
            params.df_angle_front,
            params.df_lambda_entropy,
            params.df_lambda_opacity,
            params.df_orient,
            params.df_lambda_smooth,
            params.df_gui,
            params.df_gui_w,
            params.df_gui_h,
            params.df_gui_radius,
            params.df_gui_fovy,
            params.df_gui_light_theta,
            params.df_gui_light_phi,
            params.df_max_spp
        ))

script_callbacks.on_app_started(dreamFusionAPI)

print("Dreamfusion API layer loaded")
