# coding=utf-8

from typing import Optional, List, Dict
import gradio as gr
import logging
from retry import retry

from ..base import BaseLLM
from modules.openai import Chatbot
from core import shared
from core import model

logger = logging.getLogger("ChatGPT")

CHATPROXY_PROXYS = [
    "https://bypass.churchless.tech/",
    "https://ai.fakeopen.com/api/",
    "https://api.pawan.krd/backend-api/",
]

MODEL_NAME = "ChatGPT"


class ChatGPT(BaseLLM):
    email: str = None
    password: str = None
    proxy: str = None

    def __init__(self):
        super().__init__()
        logger.info(f"Model {MODEL_NAME} init")

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    def unload_model(self):
        pass

    @retry(tries=3, delay=1)
    def reload_model(self):
        config = shared.opts.get(MODEL_NAME, None)
        logger.info(f"model config: {config}")
        self.email = config.get("email", None)
        self.password = config.get("password", None)
        self.proxy = config.get("proxy", None)

        self.unload_model()

    def generateAnswer(
        self, prompt: str, history: List[List[str]] = [], streaming: bool = False
    ) -> List[List[str]]:
        chatgptBot = Chatbot(
            config={
                "email": self.email,
                "password": self.password,
            },
            base_url=self.proxy,
        )
        if streaming:
            history += [[]]
            for data in chatgptBot.ask(prompt):
                history[-1] = [prompt, data["message"]]
                yield history
        else:
            for data in chatgptBot.ask(prompt):
                response = data["message"]
            history += [[prompt, response]]
            yield history

    def create_config_ui(self):
        with gr.Box(elem_id="chagpt_free_mode"):
            gr.Markdown(
                """
## ChatGPT免费版
使用ChatGPT免费代理来实现网页版免费聊天的效果。

"""
            )
            email = gr.Textbox(
                label="邮箱",
                info="OpenAI邮箱地址",
                value=lambda: shared.opts.get(MODEL_NAME, "email"),
            )
            password = gr.Textbox(
                label="密码",
                info="OpenAI密码",
                value=lambda: shared.opts.get(MODEL_NAME, "password"),
            )
            proxy = gr.Dropdown(
                choices=CHATPROXY_PROXYS,
                value=lambda: shared.opts.get(MODEL_NAME, "proxy"),
                label="ChatGPT代理",
                info="ChatGPT免费代理的URL",
            )
        model_config_save_btn = gr.Button(
            "保存并加载",
            elem_id="model_config_save",
            variant="primary",
        )

        def save_model_config(
            email, password, proxy, progress=gr.Progress(track_tqdm=True)
        ):
            shared.opts.set(MODEL_NAME, "email", email)
            shared.opts.set(MODEL_NAME, "password", password)
            shared.opts.set(MODEL_NAME, "proxy", proxy)
            model.reload_model(MODEL_NAME)
            return gr.update(), gr.update(), gr.update()

        response = model_config_save_btn.click(
            fn=save_model_config,
            inputs=[email, password, proxy],
            outputs=[email, password, proxy],
        )
        return response
