from typing import List, Dict


class Model(object):
    @staticmethod
    def init():
        pass

    @staticmethod
    def release():
        pass

    @staticmethod
    def chat(history: List[List[str]]):
        pass

    @staticmethod
    def stream_chat(history: List[List[str]]):
        pass
