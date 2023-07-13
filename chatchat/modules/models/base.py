# coding=utf-8

from abc import ABC, abstractmethod
from typing import List, Dict

from pydantic import BaseModel


class BaseLLM(BaseModel, ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """return name of llm."""

    @abstractmethod
    def reload_model(self):
        """reload model of llm."""

    @abstractmethod
    def unload_model(self):
        """unload model of llm."""

    @abstractmethod
    def generateAnswer(
        self, prompt: str, history: List[List[str]] = [], streaming: bool = False
    ) -> List[List[str]]:
        """return answer of llm."""

    @abstractmethod
    def create_config_ui(self):
        """return config ui of llm."""
