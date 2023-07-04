import gradio as gr
import logging

from core.models.info import llm_model_dict
from core import shared
from core.models import loader


MODE_CHAT = "对话"
MODE_CONFIG = "配置"


def mode_change(mode_choice):
    if mode_choice == MODE_CHAT:
        return gr.update(visible=True), gr.update(visible=False)
    elif mode_choice == MODE_CONFIG:
        return gr.update(visible=False), gr.update(visible=True)


def save_config(model_choice):
    current_llm_model = shared.conf["current_llm_model"]
    if model_choice != current_llm_model:
        logging.info(
            f"start to change model from {current_llm_model} to {model_choice}"
        )
        shared.conf["current_llm_model"] = model_choice
        loader.unload_model()
        loader.load_model(model_choice)


default_theme_args = dict(
    font=["Source Sans Pro", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["IBM Plex Mono", "ui-monospace", "Consolas", "monospace"],
)


def create_ui():
    with open("core/ui.css", "r") as f:
        block_css = f.read()

    with gr.Blocks(
        css=block_css,
        theme=gr.themes.Default(**default_theme_args),
    ) as app:
        with gr.Row(visible=True, elem_id="main"):
            with gr.Column(scale=2, min_width=250, elem_id="main_sider"):
                with gr.Group(elem_id="new_chat_wrap"):
                    new_chat = gr.Button(
                        value="+  新建对话", interactive=True, elem_id="new_chat"
                    )
                with gr.Group(elem_id="chat_history"):
                    gr.Markdown()
                mode_choice = gr.Radio(
                    [MODE_CHAT, MODE_CONFIG],
                    elem_id="mode_choice",
                    show_label=False,
                    value=MODE_CHAT,
                    interactive=True,
                    container=False,
                )
            with gr.Column(
                scale=13, elem_id="main_content", visible=True
            ) as main_content:
                with gr.Group(elem_id="main_content_header"):
                    gr.Markdown(
                        """
                    **Model:** ChatGPT
                    """,
                        elem_id="model_info",
                    )
                chatbot = gr.Chatbot(
                    show_label=False,
                    elem_id="chatbot",
                )
                input = gr.Textbox(
                    show_label=False,
                    container=False,
                    elem_id="input_box",
                    placeholder="Send a message...",
                )
            with gr.Column(
                scale=13, elem_id="main_config", visible=False
            ) as main_config:
                with gr.Column():
                    with gr.Tab("模型", elem_id="model_config"):
                        model_choice = gr.Dropdown(
                            list(llm_model_dict.keys()),
                            label="模型选择",
                            elem_id="model_choice",
                            value=shared.conf["current_llm_model"],
                            interactive=True,
                        )
                    with gr.Tab("通用", elem_id="app_config"):
                        gr.Markdown()
                    save_config_btn = gr.Button(
                        "保存配置", elem_id="save_config", variant="primary"
                    )
                save_config_btn.click(fn=save_config, inputs=[model_choice], outputs=[])

            mode_choice.change(
                fn=mode_change,
                inputs=[mode_choice],
                outputs=[main_content, main_config],
            )

        response = input.submit(
            lambda user_message, history: (
                gr.update(value="", interactive=False),
                history + [[user_message, None]],
            ),
            [input, chatbot],
            [input, chatbot],
            queue=False,
        )
        response.then(getattr(shared.model, "stream_chat"), chatbot, chatbot)
        response.then(lambda: gr.update(interactive=True), None, [input], queue=False)

    return app
