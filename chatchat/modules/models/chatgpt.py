from typing import Optional, List, Dict
import gradio as gr
import logging

from .base import BaseModel
from revChatGPT.V1 import Chatbot

logger = logging.getLogger("ChatGPT")


class ChatGPT(BaseModel):
    chatgptBot: object = None
    email: str = None
    password: str = None

    def __init__(self, config: dict = None):
        super().__init__()
        self.email = config.get("email", None)
        self.password = config.get("password", None)

    @property
    def _llm_type(self) -> str:
        return "ChatGPT"

    def unload_model(self):
        del self.chatgptBot
        self.chatgptBot = None

    def reload_model(self):
        self.unload_model()
        self.chatgptBot = Chatbot(
            config={
                "email": self.email,
                "password": self.password,
            },
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        logger.debug(f"__call: {prompt}")
        for data in self.chatgptBot.ask(prompt):
            response = data["message"]
        logger.debug(f"response: {response}")
        return response

    def generateAnswer(
        self, prompt: str, history: List[List[str]] = [], streaming: bool = False
    ):
        if streaming:
            history += [[]]
            for data in self.chatgptBot.ask(prompt):
                history[-1] = [prompt, data["message"]]
                yield history
        else:
            for data in self.chatgptBot.ask(prompt):
                response = data["message"]
            history += [[prompt, response]]
            yield history

    def create_config_ui(self):
        email = gr.Textbox(value=self.email)
        password = gr.Textbox(value=self.password)
        return [email, password]
