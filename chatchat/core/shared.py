# coding=utf-8

from typing import Any, Dict
import json
import logging
from logging import Logger

from core.const import *
from modules.models import BaseLLM
from modules.plugins import BasePlugin

import gradio as gr

logger: Logger = None
app: gr.Blocks = None
llm_models: Dict[str, BaseLLM] = {}
cur_llm_model_name: str = None

plugins: Dict[str, BasePlugin] = {}
cur_plugin_name: str = "None"


def get_model() -> BaseLLM:
    return llm_models[cur_llm_model_name]


def get_plugin() -> BasePlugin:
    return plugins[cur_plugin_name]


class Options:
    data = {}  # record key value
    config_file = None
    display_name_map = {"空": "None"}

    def __init__(self):
        pass

    def get(self, scope: str, key: str) -> Any:
        if scope is None:
            return None
        if key is None and scope in self.data:
            return self.data[scope]
        if scope in self.data and key in self.data[scope]:
            return self.data[scope][key]
        return None

    def set(self, scope: str, key: str, value: Any):
        if scope is None or key is None:
            return
        need_save = False
        old_value = self.get(scope, key)
        if old_value != value:
            need_save = True
        self.data[scope][key] = value
        if need_save:
            logging.info(
                f"config change, start save to file, key: {key}, old_value: {old_value}, new_value: {value}"
            )
            self.save()

    def save(self):
        with open(self.config_file, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)

    def load(self, config_file):
        self.config_file = config_file
        with open(config_file, "r", encoding="utf8") as file:
            self.data = json.load(file)

    def to_display_name(self, name):
        if name == "None":
            return "空"
        display_name = self.data[name]["display_name"]
        self.display_name_map[display_name] = name
        return display_name

    def from_display_name(self, display_name):
        return self.display_name_map[display_name]


opts = Options()
