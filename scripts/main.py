import gradio as gr

from extensions.stable_dreamfusion_extension.dreamfusion import conversion#, dreamfusion
from extensions.stable_dreamfusion_extension.dreamfusion.dreamfusion import get_df_models
from modules import script_callbacks, sd_models, shared
from modules.ui import setup_progressbar, gr_show
from webui import wrap_gradio_gpu_call

def on_ui_tabs():
    with gr.Blocks() as dreamfusion_interface:
        with gr.Row(equal_height=True):
            df_model_dir = gr.Dropdown(label='Model', choices=sorted(get_df_models()))
            
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

    return (dreamfusion_interface, "Dreamfusion", "dreamfusion_interface")

script_callbacks.on_ui_tabs(on_ui_tabs)
