from typing import List, Dict
import logging

from core.models.model import Model
from core import shared

from transformers import AutoTokenizer, AutoModel

tokenizer, model = None, None


class ChatGLM(Model):
    @staticmethod
    def init():
        global tokenizer, model
        model_name = sha
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm2-6b", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "THUDM/chatglm2-6b", trust_remote_code=True
        ).cuda()
        # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
        # from utils import load_model_on_gpus
        # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
        model.eval()
        logging.info(f"ChatGLM init done")

    @staticmethod
    def release():
        global tokenizer, model
        tokenizer, model = None, None

    @staticmethod
    def chat(history: List[List[str]]):
        pass

    @staticmethod
    def stream_chat(history: List[List[str]]):
        user_message = history[-1][0]
        history[-1][1] = ""
        old_history = history.pop(-1)
        prev_text = ""
        for new_response, _ in model.stream_chat(tokenizer, user_message, old_history):
            bot_message = new_response[len(prev_text) :]
            history[-1][1] += bot_message
            prev_text = new_response
            yield history
