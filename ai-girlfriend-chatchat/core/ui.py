import gradio as gr
import logging

from core import model
from core import shared
from core import config


MODE_CHAT = "ChatBot"
MODE_CONFIG = "Config"


def mode_change(mode_choice):
    if mode_choice == MODE_CHAT:
        return gr.update(visible=True), gr.update(visible=False)
    elif mode_choice == MODE_CONFIG:
        return gr.update(visible=False), gr.update(visible=True)


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
            with gr.Column(elem_id="main_sider"):
                with gr.Group(elem_id="new_chat_wrap"):
                    new_chat = gr.Button(
                        value="+  New Chat", interactive=True, elem_id="new_chat"
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
            with gr.Column(elem_id="main_content", visible=True) as main_content:
                with gr.Group(elem_id="main_content_header"):
                    model_info = gr.Markdown(
                        value="Model: " + shared.cur_llm_model_name,
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
                response = input.submit(
                    lambda input, chatbot: (
                        gr.update(value="", interactive=False),
                        chatbot + [[input, None]],
                    ),
                    [input, chatbot],
                    [input, chatbot],
                    queue=False,
                )
                response.then(model.stream_chat, [chatbot], [chatbot])
                response.then(
                    lambda: gr.update(interactive=True), None, [input], queue=False
                )
            with gr.Column(elem_id="main_config", visible=False) as main_config:
                with gr.Column():
                    with gr.Tab("Model", elem_id="model_config"):
                        model_choice = gr.Dropdown(
                            list(shared.llm_models.keys()),
                            show_label=False,
                            container=False,
                            label="Model Choice",
                            elem_id="model_choice",
                            value=shared.cur_llm_model_name,
                            interactive=True,
                        )
                        model_configs = []
                        components = []
                        for llm_model in shared.llm_models.keys():
                            with gr.Box(
                                elem_id="model_config_" + llm_model,
                                visible=llm_model == shared.cur_llm_model_name,
                            ) as model_config:
                                components.extend(
                                    shared.llm_models[llm_model].create_config_ui()
                                )
                            model_configs.append(model_config)
                        response = model_choice.change(
                            fn=model.model_change,
                            inputs=[model_choice],
                            outputs=model_configs + [model_info, chatbot],
                            queue=True,
                            show_progress="full",
                        )
                        response.then(
                            lambda model_choice: model.reload_model(model_choice),
                            [model_choice],
                            [],
                        )
                    with gr.Tab("System", elem_id="system_config"):
                        gr.Markdown()
            mode_choice.change(
                fn=mode_change,
                inputs=[mode_choice],
                outputs=[main_content, main_config],
            )

    return app
