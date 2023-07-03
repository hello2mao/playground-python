from typing import List, Dict


class Model(object):
    def __init__(self, config: Dict) -> None:
        pass

    def chat(self, query: str, history: List[List[str]], **kwargs):
        pass

    def stream_chat(self, query: str, history: List[List[str]], **kwargs):
        pass
