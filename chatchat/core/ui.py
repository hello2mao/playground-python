from revChatGPT.V1 import Chatbot
import gradio as gr
import os
import logging

block_css = """
#chatbot .user {
    text-align: right
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


def create_ui():
    chatgptBot = Chatbot(
        config={"email": "ihbtxrmhwqwslzk4hy@cxxx8.icu", "password": "HezbfYTBO"},
    )

    app = gr.Blocks(
        css=block_css,
        theme=gr.themes.Default(**default_theme_args),
    )
    with app:
        with gr.Row(visible=True, elem_id="main_content"):
            with gr.Column(scale=2, min_width=200):
                with gr.Accordion("See Details"):
                    gr.Markdown("lorem ipsum")
                gr.Markdown(
                    """
                        ## 提问举例:
                        1. 介绍下冲量在线
                        2. waterwheel是什么
                        3. 冲量在线可信AI一体机有什么特点
                        4. 介绍下陈浩栋
                        """
                )
            with gr.Column(scale=13):
                chatbot = gr.Chatbot(
                    show_label=False,
                    elem_id="chatbot",
                    height=500,
                )
                input = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Send a message...",
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
