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


class OptionInfo:
    def __init__(
        self,
        default=None,
        component=None,
        component_args=None,
        onchange=None,
    ):
        self.default = default
        self.component = component
        self.component_args = component_args
        self.onchange = onchange


system_info_default = {}
system_info_default.update(
    "logLevel",
    OptionInfo(
        "debug",
        gr.Textbox,
        {
            "label": "Log Level",
        },
    ),
)


class Options:
    data = {}  # record key value
    model_info = {}  # record model option info
    system_info = system_info_default  # record system option info
    config_file = None

    def __init__(self):
        self.data = {k: v.default for k, v in self.system_info.items()}

    def __setattr__(self, key: str, value: Any):
        self.data[key] = value
        return

    def __getattribute__(self, key: str) -> Any:
        if key in self.data:
            return self.data[key]
        if key in self.model_info:
            return self.model_info[key].default
        if key in self.system_info:
            return self.system_info[key].default

    def add_model_option(self, key: str, info: OptionInfo):
        self.model_info[key] = info

    def save(self):
        with open(self.config_file, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4)

    def load(self, config_file):
        self.config_file = config_file
        with open(config_file, "r", encoding="utf8") as file:
            self.data = json.load(file)


opts = Options()
