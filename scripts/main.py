import gradio as gr

from extensions.stable_dreamfusion_extension.dreamfusion import conversion, dreamfusion
from extensions.stable_dreamfusion_extension.dreamfusion.dreamfusion import get_df_models
from modules import script_callbacks, sd_models, shared
from modules.ui import setup_progressbar, gr_show
from webui import wrap_gradio_gpu_call

def on_ui_tabs():
    with gr.Blocks() as dreamfusion_interface:
        with gr.Row(equal_height=True):
            df_model_dir = gr.Dropdown(label="Model", choices=sorted(get_df_models()))
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
                        gr.HTML("test")
                        df_text = gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_negative = gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_o= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_o2= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_test= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_save_mesh= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_eval_interval= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_workspace= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_guidance= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_seed= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_iters= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_lr= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_ckpt= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_cuda_ray= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_max_steps= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_num_steps= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_upsample_steps= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_update_extra_interval= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_max_ray_batch= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_albedo= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_albedo_iters= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_uniform_sphere_rate= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_bg_radius= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_density_thresh= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_fp16= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_backbone= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_w= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_h= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_jitter_pose= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_bound= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_dt_gamma= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_min_near= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_radius_range= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_fovy_range= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_dir_text= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_suppress_face= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_angle_overhead= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_angle_front= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_lambda_entropy= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_lambda_opacity= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_orient= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_lambda_smooth= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_gui= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_gui_w= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_gui_h= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_gui_radius= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_gui_fovy= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_gui_light_theta= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_gui_light_phi= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                        df_max_spp= gr.Textbox(label='Dataset Directory', placeholder="Path to directory with input images")
                                
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

    return (dreamfusion_interface, "Dreamfusion", "dreamfusion_interface")

script_callbacks.on_ui_tabs(on_ui_tabs)
