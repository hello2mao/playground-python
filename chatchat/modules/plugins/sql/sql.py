# coding=utf-8

import logging
from typing import Any, List, Mapping, Optional

import gradio as gr

from ..base import BasePlugin
from modules.models import BaseLLM

PLUGIN_NAME = "SQL"
logger = logging.getLogger(PLUGIN_NAME)


class SQL(BasePlugin):
    def __init__(self) -> None:
        super().__init__()
        logger.info(f"Plugin {PLUGIN_NAME} init")

    @property
    def plugin_name(self) -> str:
        return PLUGIN_NAME

    def generatePluginAnswer(
        self,
        llm: BaseLLM,
        prompt: str,
        history: List[List[str]] = [],
        llm_history: List[List[str]] = [],
        streaming: bool = False,
    ):
        pass

    def create_plugin_ui(self):
        btn = gr.Button()

    def create_config_ui(self):
        pass
