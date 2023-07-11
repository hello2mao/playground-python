import logging, os
import gradio as gr
from typing import List
import tqdm

from modules.models import ChatGLM
from modules.models import ChatGPT
from modules.models import BaseModel
from modules.utils.log import record_log
from core import shared


@record_log
def init_models():
    for model_name in shared.conf["llm_models"].keys():
        class_object: BaseModel = globals().get(model_name) or locals().get(model_name)
        if class_object is None:
            logging.error(f"init_models failed: class_object is None")
            continue
        shared.llm_models[model_name] = class_object(
            shared.conf["llm_models"][model_name]
        )


@record_log
def reload_model(model_name: str):
    shared.llm_models[model_name].reload_model()
    shared.cur_llm_model_name = model_name


@record_log
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


def model_change(model_choice, progress=gr.Progress(track_tqdm=True)):
    results = []
    progress(0.2, desc="Clean Env")
    for key in list(shared.llm_models.keys()):
        results.append(gr.update(visible=key == model_choice))
    progress(0.5, desc="Update Info")
    progress(0.8, desc="Reload Model")
    return results + [gr.update(value="Model: " + model_choice), []]
