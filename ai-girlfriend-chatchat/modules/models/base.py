from abc import ABC, abstractmethod
from typing import List, Dict

from langchain.llms.base import LLM


class BaseModel(LLM, ABC):
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
