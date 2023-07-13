# coding=utf-8

from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional

from modules.models import BaseLLM

from pydantic import BaseModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class BasePlugin(BaseModel, ABC):
    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """return name of plugin."""

    @abstractmethod
    def generatePluginAnswer(
        self,
        llm: BaseLLM,
        prompt: str,
        history: List[List[str]] = [],
        llm_history: List[List[str]] = [],
        streaming: bool = False,
    ):
        """return answer of plugin."""

    @abstractmethod
    def create_plugin_ui(self):
        """return plugin ui of plugin."""

    @abstractmethod
    def create_config_ui(self):
        """return config ui of plugin."""


class LangchainLLM(LLM):
    llm: BaseLLM = None

    def __init__(self, llm: BaseLLM) -> None:
        super().__init__()
        self.llm = llm

    @property
    def _llm_type(self) -> str:
        return self.llm._model_name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        for response in self.llm.generateAnswer(
            prompt=prompt, history=[], streaming=False
        ):
            answer = response[-1][1]
        return answer
