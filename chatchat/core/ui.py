# coding=utf-8

import gradio as gr
import logging

from core import model
from core import plugin
from core import shared
from core.const import *


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
                with gr.Group(elem_id="plugin_choice_wrap"):
                    plugin_choice = gr.Dropdown(
                        choices=[
                            shared.opts.to_display_name(choice)
                            for choice in shared.opts.get(SYSTEM_CONFIG, PLUGINS)
                            + ["None"]
                        ],
                        container=False,
                        label="插件:",
                        elem_id="plugin_choice",
                        value=lambda: shared.opts.to_display_name(
                            shared.cur_plugin_name
                        ),
                        interactive=True,
                    )
                with gr.Group(elem_id="plugin_ui"):
                    plugin_uis = []
                    for plugin_name in shared.opts.get(SYSTEM_CONFIG, PLUGINS):
                        with gr.Column(
                            elem_id="plugin_ui_" + plugin_name,
                            visible=plugin_name == shared.cur_plugin_name,
                        ) as plugin_ui:
                            shared.plugins[plugin_name].create_plugin_ui()
                        plugin_uis.append(plugin_ui)
                    plugin_choice.change(
                        fn=plugin.plugin_change,
                        inputs=[plugin_choice],
                        outputs=plugin_uis,
                    )
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
                        value=lambda: "语言模型: "
                        + shared.opts.to_display_name(shared.cur_llm_model_name),
                        elem_id="model_info",
                    )
                chatbot = gr.Chatbot(
                    show_label=False,
                    elem_id="chatbot",
                )
                llm_history = gr.State(value=[])
                input = gr.Textbox(
                    show_label=False,
                    container=False,
                    elem_id="input_box",
                    placeholder="发送消息...",
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
                response.then(
                    model.stream_chat, [chatbot, llm_history], [chatbot, llm_history]
                )
                response.then(
                    lambda: gr.update(interactive=True), None, [input], queue=False
                )
            with gr.Column(elem_id="main_config", visible=False) as main_config:
                with gr.Column():
                    with gr.Tab("语言模型", elem_id="model_config"):
                        model_choice = gr.Dropdown(
                            choices=[
                                shared.opts.to_display_name(choice)
                                for choice in shared.opts.get(SYSTEM_CONFIG, LLM_MODELS)
                            ],
                            show_label=False,
                            container=False,
                            label="模型选择",
                            elem_id="model_choice",
                            value=lambda: shared.opts.to_display_name(
                                shared.cur_llm_model_name
                            ),
                            interactive=True,
                        )
                        model_configs = []
                        for llm_model in shared.opts.get(SYSTEM_CONFIG, LLM_MODELS):
                            with gr.Column(
                                elem_id="model_config_" + llm_model,
                                visible=llm_model == shared.cur_llm_model_name,
                            ) as model_config:
                                model_config_response = shared.llm_models[
                                    llm_model
                                ].create_config_ui()
                                model_config_response.then(
                                    fn=model.model_config_save,
                                    inputs=[],
                                    outputs=model_info,
                                )
                            model_configs.append(model_config)
                        model_choice.change(
                            fn=model.model_change,
                            inputs=[model_choice],
                            outputs=model_configs,
                        )
                    with gr.Tab("插件", elem_id="plugin_config"):
                        plugin_choice = gr.Dropdown(
                            choices=[
                                shared.opts.to_display_name(choice)
                                for choice in shared.opts.get(SYSTEM_CONFIG, PLUGINS)
                            ],
                            show_label=False,
                            container=False,
                            label="插件选择",
                            elem_id="plugin_choice",
                            value=lambda: shared.opts.to_display_name(
                                shared.opts.get(SYSTEM_CONFIG, PLUGINS)[0]
                            ),
                            interactive=True,
                        )
                        plugin_configs = []
                        for index, plugin_name in enumerate(
                            shared.opts.get(SYSTEM_CONFIG, PLUGINS)
                        ):
                            with gr.Column(
                                elem_id="plugin_config_" + plugin_name,
                                visible=index == 0,
                            ) as plugin_config:
                                shared.plugins[plugin_name].create_config_ui()
                            plugin_configs.append(plugin_config)
                        plugin_choice.change(
                            fn=plugin.plugin_config_change,
                            inputs=[plugin_choice],
                            outputs=plugin_configs,
                        )
                    with gr.Tab("系统", elem_id="system_config"):
                        gr.Markdown()
                        system_config_save_btn = gr.Button(
                            "保存并加载",
                            elem_id="system_config_save",
                            variant="primary",
                        )
                        response = system_config_save_btn.click(
                            fn=model.system_config_save,
                            inputs=[],
                            outputs=[],
                        )
            mode_choice.change(
                fn=mode_change,
                inputs=[mode_choice],
                outputs=[main_content, main_config],
            )

    logging.info("create ui done")
    return app
