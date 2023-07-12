# coding=utf-8

import logging
from typing import Any, List, Mapping, Optional

import gradio as gr
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

from .base import BasePlugin
from .base import LangchainLLM
from modules.models import BaseLLM
from modules.textsplitter import textsplitter

PLUGIN_NAME = "Summarization"
TRIGGER_WORDS = {
    "run": """Write a concise summary of the following:


{text}


CONCISE SUMMARY IN ITALIAN:""",
    "运行": """写出以下内容的简洁摘要:


{text}


中文的简洁摘要:""",
}
logger = logging.getLogger(PLUGIN_NAME)


class Summarization(BasePlugin):
    docs: List[Document] = None

    def __init__(self) -> None:
        super().__init__()
        logger.info(f"Plugin {PLUGIN_NAME} init")

    @property
    def _plugin_name(self) -> str:
        return PLUGIN_NAME

    def generatePluginAnswer(
        self,
        llm: BaseLLM,
        prompt: str,
        history: List[List[str]] = [],
        streaming: bool = False,
    ) -> List[List[str]]:
        if self.docs is None:
            history.append([prompt, f"文档还没解析完成，请稍等"])
        else:
            if prompt in TRIGGER_WORDS:
                langchainLLM = LangchainLLM(llm)
                PROMPT = PromptTemplate(
                    template=TRIGGER_WORDS[prompt], input_variables=["text"]
                )
                chain = load_summarize_chain(
                    langchainLLM, chain_type="stuff", prompt=PROMPT
                )
                response = chain.run(self.docs)
                history.append([prompt, response])
            else:
                history.append([prompt, f"请使用聊天触发词: 运行 或者 run"])
        yield history

    def create_plugin_ui(self):
        gr.Markdown(
            """
            插件功能：自动生成文档的摘要。
            支持的文档格式: txt、md、docx、pdf、png、jpg、jpeg、csv。

            用以下聊天词来触发摘要的功能：
            * **run** 用于生成英文版摘要。
            * **运行** 用于生成中文版摘要。
            """
        )
        file = gr.File(
            label="文件",
            file_types=[
                ".txt",
                ".md",
                ".docx",
                ".pdf",
                ".png",
                ".jpg",
                ".jpeg",
                ".csv",
            ],
            file_count="single",
        )

        def file_upload(file_obj, progress=gr.Progress(track_tqdm=True)):
            logger.debug(f"start textsplitter docs: {file_obj.name}")
            self.docs = textsplitter.split_text(filepath=file_obj.name)
            logger.debug(f"textsplitter docs done, docs len: {len(self.docs)}")

        file.upload(fn=file_upload, inputs=[file], outputs=None)

    def create_config_ui(self):
        pass
