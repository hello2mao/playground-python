import gradio as gr
import logging

from core import model
from core import shared


def create_ui():
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
    response.then(lambda: gr.update(interactive=True), None, [input], queue=False)
