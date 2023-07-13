# coding=utf-8

import logging, os
import gradio as gr
from typing import List
from retry import retry

from modules.models import ChatGLM
from modules.models import ChatGPT
from modules.models import Baichuan
from modules.models import BaseLLM
from modules.utils.log import record_log
from core import shared
from core.const import *


@record_log
def init_models():
    for model_name in shared.opts.get(SYSTEM_CONFIG, LLM_MODELS):
        class_object: BaseLLM = globals().get(model_name) or locals().get(model_name)
        if class_object is None:
            logging.error(f"init_models failed: class_object is None")
            os._exit(-1)
        shared.llm_models[model_name] = class_object()


@record_log
def reload_model(model_name: str):
    try:
        shared.llm_models[model_name].reload_model()
    except Exception as err:
        errMsg = f"reload_model to {model_name} failed: {err}"
        logging.error(errMsg)
        raise gr.Error(errMsg)
    shared.cur_llm_model_name = model_name
    shared.opts.set(SYSTEM_CONFIG, DEFAULT_LLM_MODEL, model_name)


@record_log
def unload_model():
    if shared.cur_llm_model_name is None:
        return
    logging.info(f"start unload_model: {shared.cur_llm_model_name}")
    shared.get_model().unload_model()
    logging.info(f"unload_model done: {shared.cur_llm_model_name}")


def stream_chat(chatbot: List[List[str]], llm_history: List[List[str]]):
    if shared.cur_llm_model_name is None:
        return
    if shared.cur_plugin_name == "None":
        try:
            input = chatbot[-1][0]
            for history in shared.get_model().generateAnswer(
                input, chatbot[:-1], streaming=True
            ):
                yield history, history
        except Exception as err:
            errMsg = f"错误：{err}"
            logging.error(errMsg)
            raise gr.Error(errMsg)
    else:
        try:
            input = chatbot[-1][0]
            for history, llm_history in shared.get_plugin().generatePluginAnswer(
                shared.get_model(), input, chatbot[:-1], llm_history, streaming=False
            ):
                yield history, llm_history
        except Exception as err:
            errMsg = f"错误：{err}"
            logging.error(errMsg)
            raise gr.Error(errMsg)


def model_change(model_choice):
    model_choice = shared.opts.from_display_name(model_choice)
    results = []
    for model_name in shared.opts.get(SYSTEM_CONFIG, LLM_MODELS):
        results.append(gr.update(visible=model_name == model_choice))
    return results


def model_config_save():
    return gr.update(
        value="Model: " + shared.opts.to_display_name(shared.cur_llm_model_name)
    )


def system_config_save():
    pass
