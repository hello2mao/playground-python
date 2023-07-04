from typing import List, Dict

from core import shared
from core.models.model import Model

from revChatGPT.V1 import Chatbot

chatgptBot = None


class ChatGPT(Model):
    @staticmethod
    def init():
        global chatgptBot
        chatgptBot = Chatbot(
            config={
                "email": shared.conf["chatgpt"]["email"],
                "password": shared.conf["chatgpt"]["password"],
            },
        )

    @staticmethod
    def release():
        global chatgptBot
        chatgptBot = None

    @staticmethod
    def chat(history: List[List[str]]):
        pass

    @staticmethod
    def stream_chat(history: List[List[str]]):
        user_message = history[-1][0]
        history[-1][1] = ""
        prev_text = ""
        for data in chatgptBot.ask(user_message):
            bot_message = data["message"][len(prev_text) :]
            history[-1][1] += bot_message
            prev_text = data["message"]
            yield history
