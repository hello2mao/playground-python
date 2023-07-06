import logging, os
import gradio as gr
from typing import List

from modules.models import ChatGLM
from modules.models import ChatGPT
from modules.models import BaseModel

from core import shared


def init_models():
    logging.info(f"start init_models")
    for model_name in shared.conf["llm_models"].keys():
        class_object: BaseModel = globals().get(model_name) or locals().get(model_name)
        if class_object is None:
            logging.error(f"init_models failed: class_object is None")
            continue
        shared.llm_models[model_name] = class_object(
            shared.conf["llm_models"][model_name]
        )
    logging.info(f"init_models done")


def reload_model(model_name: str):
    logging.info(f"start reload_model: {model_name}")
    shared.llm_models[model_name].reload_model()
    shared.cur_llm_model_name = model_name
    logging.info(f"reload_model done: {model_name}")


def unload_model():
    if shared.cur_llm_model_name is None:
        return
    logging.info(f"start unload_model: {shared.cur_llm_model_name}")
    shared.get_model().unload_model()
    logging.info(f"unload_model done: {shared.cur_llm_model_name}")


def stream_chat(chatbot: List[List[str]]):
    if shared.cur_llm_model_name is None:
        return
    input = chatbot[-1][0]
    for history in shared.get_model().generateAnswer(
        input, chatbot[:-1], streaming=True
    ):
        yield history


def model_change(model_choice):
    results = []
    for key in list(shared.llm_models.keys()):
        results.append(gr.update(visible=key == model_choice))
    return results
