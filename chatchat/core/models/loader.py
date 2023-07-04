import logging
import os

from core import shared
from core.models.info import *
from core.models.model import Model
from core.models.chatglm import ChatGLM
from core.models.chatgpt import ChatGPT


def load_model(model_name: str):
    model_info = llm_model_dict[model_name]
    if model_info is None:
        logging.error(f"model init failed: model_info is None")
        os._exit(-1)
    model_class = model_info["class"]
    logging.info(f"model init, model_class: {model_class}")
    class_object: Model = globals().get(model_class) or locals().get(model_class)
    if class_object is None:
        logging.error(f"model init failed: class_object is None")
        os._exit(-1)
    shared.model = class_object

    init_method = getattr(class_object, "init")
    init_method()


def unload_model():
    release = getattr(shared.model, "release")
    release()
