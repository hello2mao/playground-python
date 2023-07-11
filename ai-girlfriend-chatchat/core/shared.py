from typing import Any, Dict
import json
import logging

from modules.models import BaseModel

import gradio as gr

app: gr.Blocks = None
cur_llm_model_name: str = None
llm_models: Dict = {}


def get_model() -> BaseModel:
    return llm_models[cur_llm_model_name]
