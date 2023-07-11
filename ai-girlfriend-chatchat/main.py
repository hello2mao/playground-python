from revChatGPT.V1 import Chatbot
import gradio as gr
import os
import logging

from core import initialize
from core import ui
from core import shared


if __name__ == "__main__":
    initialize.initialize()

    shared.app = ui.create_ui()

    shared.app.queue(concurrency_count=3)
    app, local_url, share_url = shared.app.launch(
        server_name="0.0.0.0",
        server_port=8077,
        debug=False,
        inbrowser=False,
        show_api=False,
        share=False,
    )
