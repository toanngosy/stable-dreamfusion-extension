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
            #df_load_params = gr.Button(value="Load Params")
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
                        df_text = gr.Textbox(label='text prompt', value='a hamburger', default='', placeholder="text prompt")
                        df_negative = gr.Textbox(label='negative', value='', default='', placeholder="negative text prompt")
                        df_o = gr.Checkbox(label='o', value=True, placeholder="equals --fp16 --cuda_ray --dir_text")
                        df_o2 = gr.Checkbox(label='o2', value=False, placeholder="equals --backbone vanilla --dir_text")
                        df_test = gr.Checkbox(label='test', value=False, placeholder="test mode")
                        df_save_mesh = gr.Checkbox(label='save_mesh', value=False, placeholder="export an obj mesh with texture")
                        df_eval_interval = gr.Number(label='eval_interval', value=10, default=10, precision=0, placeholder="evaluate on the valid set every interval epochs")
                        df_workspace = gr.Textbox(label='workspace', value='trial', default='workspace', placeholder="workspace")
                        df_guidance = gr.Textbox(label='guidance', value='stable-diffusion', default='stable-diffusion', placeholder="choose from [stable-diffusion, clip]")
                        df_seed = gr.Number(label='seed', value=0, default=0, precision=0, placeholder="seed")
                        df_iters = gr.Number(label='iters', value=10000, default=10000, precision=0, placeholder="training iters")
                        df_lr = gr.Number(label='lr', value=1e-3, default=1e-3, placeholder="initial learning rate")
                        df_ckpt = gr.Textbox(label='ckpt', value='latest', default='latest', placeholder="ckpt")
                        df_cuda_ray = gr.Checkbox(label='cuda_ray', value=False, placeholder="use CUDA raymarching instead of pytorch")
                        df_max_steps = gr.Number(label='max_steps', value=512, default=512, precision=0, placeholder="max num steps sampled per ray (only valid when using --cuda_ray)")
                        df_num_steps = gr.Number(label='num_steps', value=64, default=64, precision=0, placeholder="num steps sampled per ray (only valid when not using --cuda_ray)")
                        df_upsample_steps = gr.Number(label='upsample_steps', value=32, default=32, precision=0, placeholder="num steps up-sampled per ray (only valid when not using --cuda_ray)")
                        df_update_extra_interval = gr.Number(label='update_extra_interval', value=0, precision=0, default=16, placeholder="iter interval to update extra status (only valid when using --cuda_ray)")
                        df_max_ray_batch = gr.Number(label='max_ray_batch', value=4096, default=4096, precision=0, placeholder="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
                        df_albedo = gr.Checkbox(label='albedo', value=False, placeholder="only use albedo shading to train, overrides --albedo_iters")
                        df_albedo_iters = gr.Number(label='albedo_iters', value=1000, default=1000, precision=0, placeholder="training iters that only use albedo shading")
                        df_uniform_sphere_rate = gr.Number(label='uniform_sphere_rate', value=0.5, default=0.5, placeholder="likelihood of sampling camera location uniformly on the sphere surface area")
                        df_bg_radius = gr.Number(label='bg_radius', value=1.4, default=1.4, placeholder="if positive, use a background model at sphere(bg_radius)")
                        df_density_thresh = gr.Number(label='density_thresh', value=10, default=10, placeholder="threshold for density grid to be occupied")
                        df_fp16 = gr.Checkbox(label='fp16',  value=False, placeholder="use amp mixed precision training")
                        df_backbone = gr.Textbox(label='backbone', value='grid', default='grid', placeholder="nerf backbone, choose from [grid, vanilla]")
                        df_w = gr.Number(label='w', value=64, default=64, precision=0, placeholder="render width for NeRF in training")
                        df_h = gr.Number(label='h', value=64, default=64, precision=0, placeholder="render height for NeRF in training")
                        df_jitter_pose = gr.Checkbox(label='jitter_pose', value=False, placeholder="add jitters to the randomly sampled camera poses")
                        df_bound = gr.Number(label='bound', value=1, default=1, placeholder="assume the scene is bounded in box(-bound, bound)")
                        df_dt_gamma = gr.Number(label='dt_gamma', value=0, default=0, placeholder="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
                        df_min_near = gr.Number(label='min_near', value=0.1, default=0.1, placeholder="minimum near distance for camera")
                        df_radius_range = gr.Textbox(label='radius_range', value=[1.0, 1.5], default=[1.0, 1.5], placeholder="training camera radius range")
                        df_fovy_range = gr.Textbox(label='fovy_range', value=[40, 70], default=[40, 70], placeholder="training camera fovy range")
                        df_dir_text = gr.Checkbox(label='dir_text', value=False, placeholder="direction-encode the text prompt, by appending front/side/back/overhead view")
                        df_suppress_face = gr.Checkbox(label='suppress_face', value=False, placeholder="also use negative dir text prompt.")
                        df_angle_overhead = gr.Number(label='angle_overhead', value=30, default=30, placeholder="[0, angle_overhead] is the overhead region")
                        df_angle_front = gr.Number(label='angle_front', value=60, default=60, placeholder="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
                        df_lambda_entropy = gr.Number(label='lambda_entropy', value=1e-4, default=1e-4, placeholder="loss scale for alpha entropy")
                        df_lambda_opacity = gr.Number(label='lambda_opacity', value=0, default=0, placeholder="loss scale for alpha value")
                        df_orient = gr.Number(label='orient', value=1e-2, default=1e-2, placeholder="loss scale for orientation")
                        df_lambda_smooth = gr.Number(label='lambda_smooth', value=0, default=0, placeholder="loss scale for surface smoothness")
                        df_gui = gr.Checkbox(label='gui', value=False, placeholder="start a GUI")
                        df_gui_w = gr.Number(label='gui_w', value=800, default=800, precision=0, placeholder="GUI width")
                        df_gui_h = gr.Number(label='gui_h', value=800, default=800, precision=0, placeholder="GUI height")
                        df_gui_radius = gr.Number(label='gui_radius', value=3, default=3, placeholder="default GUI camera radius from center")
                        df_gui_fovy = gr.Number(label='gui_fovy', value=60, default=60, placeholder="default GUI camera fovy")
                        df_gui_light_theta = gr.Number(label='gui_light_theta', value=60, default=60, placeholder="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
                        df_gui_light_phi = gr.Number(label='gui_light_phi', value=0, default=0, placeholder="default GUI light direction in [0, 360), azimuth")
                        df_max_spp = gr.Number(label='max_spp', value=1, default=1, placeholder="GUI rendering max sample per pixel")
                                
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

        # df_load_params.click(
        #     fn=dreamfusion.load_params,
        #     inputs=[
        #         df_model_dir
        #     ],
        #     outputs=[
        #         df_model_dir,
        #         df_half_model,
        #         df_text,
        #         df_negative,
        #         df_o,
        #         df_o2,
        #         df_test,
        #         df_save_mesh,
        #         df_eval_interval,
        #         df_workspace,
        #         df_guidance,
        #         df_seed,
        #         df_iters,
        #         df_lr,
        #         df_ckpt,
        #         df_cuda_ray,
        #         df_max_steps,
        #         df_num_steps,
        #         df_upsample_steps,
        #         df_update_extra_interval,
        #         df_max_ray_batch,
        #         df_albedo,
        #         df_albedo_iters,
        #         df_uniform_sphere_rate,
        #         df_bg_radius,
        #         df_density_thresh,
        #         df_fp16,
        #         df_backbone,
        #         df_w,
        #         df_h,
        #         df_jitter_pose,
        #         df_bound,
        #         df_dt_gamma,
        #         df_min_near,
        #         df_radius_range,
        #         df_fovy_range,
        #         df_dir_text,
        #         df_suppress_face,
        #         df_angle_overhead,
        #         df_angle_front,
        #         df_lambda_entropy,
        #         df_lambda_opacity,
        #         df_orient,
        #         df_lambda_smooth,
        #         df_gui,
        #         df_gui_w,
        #         df_gui_h,
        #         df_gui_radius,
        #         df_gui_fovy,
        #         df_gui_light_theta,
        #         df_gui_light_phi,
        #         df_max_spp
        #     ]
        # )

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
