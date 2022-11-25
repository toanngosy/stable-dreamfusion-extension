import gradio as gr

from extensions.stable_dreamfusion_extension.dreamfusion import conversion, dreamfusion
from extensions.stable_dreamfusion_extension.dreamfusion.dreamfusion import get_df_models
from modules import script_callbacks, sd_models, shared
from modules.ui import setup_progressbar, gr_show
from webui import wrap_gradio_gpu_call

def on_ui_tabs():
    with gr.Blocks() as dreamfusion_interface:
        with gr.Row(equal_height=True):
            df_model_dir = gr.Dropdown(label='Model', choices=sorted(get_df_models()))
            df_half_model = gr.Checkbox(label="Half", value=False)
            df_load_params = gr.Button(value="Load Params")
            df_train_embedding = gr.Button(value="Train", variant="primary")

        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel"):
                with gr.Tab("Create Model"):
                    df_new_model_name = gr.Textbox(label="Name")
                    df_create_from_hub = gr.Checkbox(label="Import Model from Huggingface Hub", value=False)
                    with gr.Column(visible=False) as hub_row:
                        df_new_model_url = gr.Textbox(label="Model Path", value="runwayml/stable-diffusion-v1-5")
                        df_new_model_token = gr.Textbox(label="HuggingFace Token", value="")
                    with gr.Row() as local_row:
                        src_checkpoint = gr.Dropdown(label="Source Checkpoint",
                                                     choices=sorted(sd_models.checkpoints_list.keys()))
                    diff_type = gr.Dropdown(label="Scheduler", choices=["ddim", "pndm", "lms"], value="ddim")
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")
                        
                        with gr.Column():
                            df_create_embedding = gr.Button(value="Create", variant="primary")
            
                with gr.Tab("Train Model"):
                    with gr.Column():
                        gr.HTML("Parameter")
                        df_text = gr.Textbox(label='text prompt', placeholder="text prompt")
                        df_negative = gr.Textbox(label='negative', placeholder="negative text prompt")
                        df_o = gr.Textbox(label='o', placeholder="equals --fp16 --cuda_ray --dir_text")
                        df_o2 = gr.Textbox(label='o2', placeholder="equals --backbone vanilla --dir_text")
                        df_test = gr.Textbox(label='test', placeholder="test mode")
                        df_save_mesh = gr.Textbox(label='save_mesh', placeholder="export an obj mesh with texture")
                        df_eval_interval = gr.Textbox(label='eval_interval', placeholder="evaluate on the valid set every interval epochs")
                        df_workspace = gr.Textbox(label='workspace', placeholder="workspace")
                        df_guidance = gr.Textbox(label='guidance', placeholder="choose from [stable-diffusion, clip]")
                        df_seed = gr.Textbox(label='seed', placeholder="seed")
                        df_iters = gr.Textbox(label='iters', placeholder="training iters")
                        df_lr = gr.Textbox(label='lr', placeholder="initial learning rate")
                        df_ckpt = gr.Textbox(label='ckpt', placeholder="latest")
                        df_cuda_ray = gr.Textbox(label='cuda_ray', placeholder="use CUDA raymarching instead of pytorch")
                        df_max_steps = gr.Textbox(label='max_steps', placeholder="max num steps sampled per ray (only valid when using --cuda_ray)")
                        df_num_steps = gr.Textbox(label='num_steps', placeholder="num steps sampled per ray (only valid when not using --cuda_ray)")
                        df_upsample_steps = gr.Textbox(label='upsample_steps', placeholder="num steps up-sampled per ray (only valid when not using --cuda_ray)")
                        df_update_extra_interval = gr.Textbox(label='update_extra_interval', placeholder="iter interval to update extra status (only valid when using --cuda_ray)")
                        df_max_ray_batch = gr.Textbox(label='max_ray_batch', placeholder="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
                        df_albedo = gr.Textbox(label='albedo', placeholder="only use albedo shading to train, overrides --albedo_iters")
                        df_albedo_iters = gr.Textbox(label='albedo_iters', placeholder="training iters that only use albedo shading")
                        df_uniform_sphere_rate = gr.Textbox(label='uniform_sphere_rate', placeholder="likelihood of sampling camera location uniformly on the sphere surface area")
                        df_bg_radius = gr.Textbox(label='bg_radius', placeholder="if positive, use a background model at sphere(bg_radius)")
                        df_density_thresh = gr.Textbox(label='density_thresh', placeholder="threshold for density grid to be occupied")
                        df_fp16 = gr.Textbox(label='fp16', placeholder="use amp mixed precision training")
                        df_backbone = gr.Textbox(label='backbone', placeholder="nerf backbone, choose from [grid, vanilla]")
                        df_w = gr.Textbox(label='w', placeholder="render width for NeRF in training")
                        df_h = gr.Textbox(label='h', placeholder="render height for NeRF in training")
                        df_jitter_pose = gr.Textbox(label='jitter_pose', placeholder="add jitters to the randomly sampled camera poses")
                        df_bound = gr.Textbox(label='bound', placeholder="assume the scene is bounded in box(-bound, bound)")
                        df_dt_gamma = gr.Textbox(label='dt_gamma', placeholder="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
                        df_min_near = gr.Textbox(label='min_near', placeholder="minimum near distance for camera")
                        df_radius_range = gr.Textbox(label='radius_range', placeholder="training camera radius range")
                        df_fovy_range = gr.Textbox(label='fovy_range', placeholder="training camera fovy range")
                        df_dir_text = gr.Textbox(label='dir_text', placeholder="direction-encode the text prompt, by appending front/side/back/overhead view")
                        df_suppress_face = gr.Textbox(label='suppress_face', placeholder="also use negative dir text prompt.")
                        df_angle_overhead = gr.Textbox(label='angle_overhead', placeholder="[0, angle_overhead] is the overhead region")
                        df_angle_front = gr.Textbox(label='angle_front', placeholder="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
                        df_lambda_entropy = gr.Textbox(label='lambda_entropy', placeholder="loss scale for alpha entropy")
                        df_lambda_opacity = gr.Textbox(label='lambda_opacity', placeholder="loss scale for alpha value")
                        df_orient = gr.Textbox(label='orient', placeholder="loss scale for orientation")
                        df_lambda_smooth = gr.Textbox(label='lambda_smooth', placeholder="loss scale for surface smoothness")
                        df_gui = gr.Textbox(label='gui', placeholder="start a GUI")
                        df_gui_w = gr.Textbox(label='gui_w', placeholder="GUI width")
                        df_gui_h = gr.Textbox(label='gui_h', placeholder="GUI height")
                        df_gui_radius = gr.Textbox(label='gui_radius', placeholder="default GUI camera radius from center")
                        df_gui_fovy = gr.Textbox(label='gui_fovy', placeholder="default GUI camera fovy")
                        df_gui_light_theta = gr.Textbox(label='gui_light_theta', placeholder="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
                        df_gui_light_phi = gr.Textbox(label='gui_light_phi', placeholder="default GUI light direction in [0, 360), azimuth")
                        df_max_spp = gr.Textbox(label='max_spp', placeholder="GUI rendering max sample per pixel")
                                
            with gr.Column(variant="panel"):
                df_status = gr.HTML(elem_id="df_status", value="")
                df_progress = gr.HTML(elem_id="df_progress", value="")
                df_outcome = gr.HTML(elem_id="df_error", value="")
                df_progressbar = gr.HTML(elem_id="df_progressbar")
                df_gallery = gr.Gallery(label='Output', show_label=False, elem_id='df_gallery').style(grid=4)
                df_preview = gr.Image(elem_id='df_preview', visible=False)
                setup_progressbar(df_progressbar, df_preview, 'df', textinfo=df_progress)


        df_create_embedding.click(
            fn=conversion.extract_checkpoint,
            inputs=[
                df_new_model_name,
                src_checkpoint,
                diff_type,
                df_new_model_url,
                df_new_model_token
            ],
            outputs=[
                df_model_dir,
                df_status,
                df_outcome,
            ]
        )
        
        df_create_from_hub.change(
            fn=lambda x: gr_show(x),
            inputs=[df_create_from_hub],
            outputs=[hub_row],
        )

        df_create_from_hub.change(
            fn=lambda x: {
                hub_row: gr_show(x is True),
                local_row: gr_show(x is False)
            },
            inputs=[df_create_from_hub],
            outputs=[
                hub_row,
                local_row
            ]
        )

        df_load_params.click(
            fn=dreamfusion.load_params,
            inputs=[
                df_model_dir
            ],
            outputs=[
                df_model_dir,
                df_half_model,
                df_text,
                df_negative,
                df_o,
                df_o2,
                df_test,
                df_save_mesh,
                df_eval_interval,
                df_workspace,
                df_guidance,
                df_seed,
                df_iters,
                df_lr,
                df_ckpt,
                df_cuda_ray,
                df_max_steps,
                df_num_steps,
                df_upsample_steps,
                df_update_extra_interval,
                df_max_ray_batch,
                df_albedo,
                df_albedo_iters,
                df_uniform_sphere_rate,
                df_bg_radius,
                df_density_thresh,
                df_fp16,
                df_backbone,
                df_w,
                df_h,
                df_jitter_pose,
                df_bound,
                df_dt_gamma,
                df_min_near,
                df_radius_range,
                df_fovy_range,
                df_dir_text,
                df_suppress_face,
                df_angle_overhead,
                df_angle_front,
                df_lambda_entropy,
                df_lambda_opacity,
                df_orient,
                df_lambda_smooth,
                df_gui,
                df_gui_w,
                df_gui_h,
                df_gui_radius,
                df_gui_fovy,
                df_gui_light_theta,
                df_gui_light_phi,
                df_max_spp
            ]
        )

        df_train_embedding.click(
            fn=wrap_gradio_gpu_call(dreamfusion.start_training, extra_outputs=[gr.update()]),
            _js="start_training_dreamfusion",
            inputs=[
                df_model_dir,
                df_half_model,
                df_text,
                df_negative,
                df_o,
                df_o2,
                df_test,
                df_save_mesh,
                df_eval_interval,
                df_workspace,
                df_guidance,
                df_seed,
                df_iters,
                df_lr,
                df_ckpt,
                df_cuda_ray,
                df_max_steps,
                df_num_steps,
                df_upsample_steps,
                df_update_extra_interval,
                df_max_ray_batch,
                df_albedo,
                df_albedo_iters,
                df_uniform_sphere_rate,
                df_bg_radius,
                df_density_thresh,
                df_fp16,
                df_backbone,
                df_w,
                df_h,
                df_jitter_pose,
                df_bound,
                df_dt_gamma,
                df_min_near,
                df_radius_range,
                df_fovy_range,
                df_dir_text,
                df_suppress_face,
                df_angle_overhead,
                df_angle_front,
                df_lambda_entropy,
                df_lambda_opacity,
                df_orient,
                df_lambda_smooth,
                df_gui,
                df_gui_w,
                df_gui_h,
                df_gui_radius,
                df_gui_fovy,
                df_gui_light_theta,
                df_gui_light_phi,
                df_max_spp
            ],
            outputs=[
                df_status,
                df_outcome,
            ]
        )
    return (dreamfusion_interface, "Dreamfusion", "dreamfusion_interface"),

script_callbacks.on_ui_tabs(on_ui_tabs)
