from typing import Dict

from modules.models import BaseModel

import gradio as gr

app: gr.Blocks = None
conf: Dict = None
cur_llm_model_name: str = None
llm_models: Dict = {}


def get_model() -> BaseModel:
    return llm_models[cur_llm_model_name]
