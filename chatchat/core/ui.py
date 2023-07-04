from revChatGPT.V1 import Chatbot
import gradio as gr
import os
import logging

block_css = """
.gap {
    gap: 0px !important;
}

.hide {
    display: none !important;
}

/* app网页全屏 */
.gradio-container {
    padding: 0 !important;
    position: none !important;
}
.app {
    max-width: 100% !important;
}
.app .contain {
    display: flex;
}
.app .contain #main {
    flex-grow: 1;
    padding: 0;
}

#main {
    padding-bottom: 25px;
}

#main_content {
    display: flex;
    align-items: center;
    padding-bottom: 25px;
    padding-right: 25px;
}

#main_content #main_content_header {
    height: 40px;
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
}

#main_content #main_content_header #model_info {
    width: 200px;
}

#main_content #model_choice {
    width: 200px;
    flex-grow: none;
}

#main_content #model_choice label {
    display: flex;
    align-items: center;
    justify-content: center;
}

#main_content #model_choice label span {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0;
    margin-right: 10px;
}

#main_content .form {
    width: 800px;
}

#main_content .gap {
    gap: 8px;
}

#chatbot {
    margin-bottom: 20px;
    border: 1px solid rgba(0, 0, 0, 0) !important;
    box-shadow: none;
    width: 800px;
}

#chatbot .user {
    text-align: right;
}

#input_box textarea {
    height: 60px !important;
    display: flex;
    justify-content: center;
    align-items: center;
}
#model_info {
    width: 800px;
}

/* main_sider */

#main_sider {
    display: flex;
    flex-direction: column;
    justify-content: start;
    align-items: center;
    border: 2px solid #f3f4f6;
}

#main_sider #new_chat_wrap {
    flex-grow: 0;
    display: flex;
    justify-content: center;
    align-items: center;
}

#main_sider #new_chat {
    margin: 10px;
    display: flex;
    justify-content: left;
    align-items: center;
    border-top-right-radius: 8px !important;
    border-top-left-radius: 8px !important;
    border-bottom-right-radius: 8px !important;
    border-bottom-left-radius: 8px !important;
    
}

#main_sider #chat_history {
    flex-grow: 1;
}

#main_sider .form {
    border-radius: 0;
    flex-grow: 0;
}

#main_sider .form #mode_choice .wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 10px;
    margin-bottom: 10px;
}



/* 隐藏Gradio底部 */
footer {
    visibility: hidden;
    margin: 0 !important;
    height: 0 !important;
}

.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}
"""

default_theme_args = dict(
    font=["Source Sans Pro", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["IBM Plex Mono", "ui-monospace", "Consolas", "monospace"],
)

MODE_CHAT = "对话"
MODE_CONFIG = "配置"


def mode_change(mode_choice):
    if mode_choice == MODE_CHAT:
        return gr.update(visible=True), gr.update(visible=False)
    elif mode_choice == MODE_CONFIG:
        return gr.update(visible=False), gr.update(visible=True)


def create_ui():
    chatgptBot = Chatbot(
        config={"email": "tcohen6@bodytracker.shop", "password": "@Ahrqg3E$j"},
    )

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
                    gr.Markdown(
                        """
                    haha
                    """
                    )
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
                model_choice = gr.Radio(
                    ["ChatGPT", "THUDM/ChatGLM2-6B"],
                    elem_id="model_choice",
                    show_label=False,
                    value="ChatGPT",
                    interactive=True,
                    container=False,
                )

            mode_choice.change(
                fn=mode_change,
                inputs=[mode_choice],
                outputs=[main_content, main_config],
            )

        def gen_response(history):
            user_message = history[-1][0]
            history[-1][1] = ""
            prev_text = ""
            for data in chatgptBot.ask(user_message):
                bot_message = data["message"][len(prev_text) :]
                history[-1][1] += bot_message
                prev_text = data["message"]
                yield history

        response = input.submit(
            lambda user_message, history: (
                gr.update(value="", interactive=False),
                history + [[user_message, None]],
            ),
            [input, chatbot],
            [input, chatbot],
            queue=False,
        )
        response.then(gen_response, chatbot, chatbot)
        response.then(lambda: gr.update(interactive=True), None, [input], queue=False)

    return app
