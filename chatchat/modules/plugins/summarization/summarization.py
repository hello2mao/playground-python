# coding=utf-8

import logging
from typing import Any, List, Mapping, Optional
import os

import gradio as gr
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

from ..base import BasePlugin
from ..base import LangchainLLM
from core import shared
from core.const import *
from modules.models import BaseLLM
from modules.textsplitter import textsplitter

PLUGIN_NAME = "Summarization"
TRIGGER_WORDS = ["摘要"]
logger = logging.getLogger(PLUGIN_NAME)


class Summarization(BasePlugin):
    docs: List[Document] = None

    # config
    chain_type: str = "stuff"
    chunk_limit: int = 10000
    file_size_limit: int = 10 * 1024 * 1024  # 10M
    sentence_size: int = 100
    using_zh_title_enhance: bool = True
    prompt_template: str = """
请给出以下内容的简洁摘要:

    
{text}


简洁的中文摘要:
"""

    def __init__(self) -> None:
        super().__init__()
        logger.info(f"Plugin {PLUGIN_NAME} init")

    @property
    def plugin_name(self) -> str:
        return PLUGIN_NAME

    def _update_config(self):
        self.chain_type = (
            shared.opts.get(
                PLUGIN_NAME,
                "chain_type",
            )
            or self.chain_type
        )
        self.chunk_limit = (
            shared.opts.get(
                PLUGIN_NAME,
                "chunk_limit",
            )
            or self.chunk_limit
        )
        self.file_size_limit = (
            shared.opts.get(
                PLUGIN_NAME,
                "file_size_limit",
            )
            or self.file_size_limit
        )
        self.sentence_size = (
            shared.opts.get(
                PLUGIN_NAME,
                "sentence_size",
            )
            or self.sentence_size
        )
        self.using_zh_title_enhance = (
            shared.opts.get(
                PLUGIN_NAME,
                "using_zh_title_enhance",
            )
            or self.using_zh_title_enhance
        )
        self.prompt_template = (
            shared.opts.get(
                PLUGIN_NAME,
                "prompt_template",
            )
            or self.prompt_template
        )

    def generatePluginAnswer(
        self,
        llm: BaseLLM,
        prompt: str,
        history: List[List[str]] = [],
        llm_history: List[List[str]] = [],
        streaming: bool = False,
    ):
        self._update_config()
        if self.docs is None:
            history.append([prompt, f"文档还没解析完成，请稍等"])
        else:
            if prompt in TRIGGER_WORDS:
                langchainLLM = LangchainLLM(llm)
                PROMPT = PromptTemplate(
                    template=self.prompt_template, input_variables=["text"]
                )
                logger.info(
                    f"start load_summarize_chain using chain: {self.chain_type}"
                )
                chain = load_summarize_chain(
                    langchainLLM, chain_type=self.chain_type, prompt=PROMPT
                )
                response = chain.run(self.docs[: min(self.chunk_limit, len(self.docs))])
                history.append([prompt, response])
            else:
                history.append([prompt, f"请使用摘要发词: {TRIGGER_WORDS}"])
        yield history, history

    def create_plugin_ui(self):
        gr.Markdown(
            """
            1.插件功能：自动生成文档的摘要。

            2.支持的文档格式: txt、md、docx、pdf、png、jpg、jpeg、csv。

            3.获取结果：在右侧输入聊天词**摘要**。

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
            file_stats = os.stat(file_obj.name)
            file_size = file_stats.st_size  # 获取文件大小（以字节为单位）
            if file_size > self.file_size_limit:
                errMsg = f"file_upload failed: file_size > {self.file_size_limit}"
                logger.error(errMsg)
                gr.Error(errMsg)
                return
            self.docs = textsplitter.split_text(
                filepath=file_obj.name,
                sentence_size=self.sentence_size,
                using_zh_title_enhance=self.using_zh_title_enhance,
            )
            logger.debug(f"textsplitter docs done, docs len: {len(self.docs)}")

        file.upload(fn=file_upload, inputs=[file], outputs=None)

    def create_config_ui(self):
        with gr.Row():
            with gr.Column():
                chain_type = gr.Dropdown(
                    label="摘要方式",
                    info="此选择会影响调用语言模型的次数，请慎重选择",
                    choices=["stuff", "refine", "map_reduce"],
                    value=lambda: shared.opts.get(PLUGIN_NAME, "chain_type")
                    or self.chain_type,
                )
                chunk_limit = gr.Number(
                    label="摘要块数上限",
                    precision=0,
                    value=lambda: shared.opts.get(PLUGIN_NAME, "chunk_limit")
                    or self.chunk_limit,
                )
                file_size_limit = gr.Number(
                    label="文档大小上限",
                    precision=0,
                    value=lambda: shared.opts.get(PLUGIN_NAME, "file_size_limit")
                    or self.file_size_limit,
                )
            with gr.Column():
                sentence_size = gr.Number(
                    label="文本分句长度",
                    precision=0,
                    value=lambda: shared.opts.get(PLUGIN_NAME, "sentence_size")
                    or self.sentence_size,
                )
                using_zh_title_enhance = gr.Checkbox(
                    label="开启中文标题加强",
                    value=lambda: shared.opts.get(PLUGIN_NAME, "using_zh_title_enhance")
                    or self.using_zh_title_enhance,
                )
        prompt_template = gr.TextArea(
            label="提示词模板",
            value=lambda: shared.opts.get(PLUGIN_NAME, "prompt_template")
            or self.prompt_template,
        )

        plugin_config_save_btn = gr.Button(
            "保存",
            elem_id="plugin_config_save",
            variant="primary",
        )

        def save_plugin_config(
            chain_type,
            chunk_limit,
            file_size_limit,
            sentence_size,
            using_zh_title_enhance,
            prompt_template,
        ):
            shared.opts.set(
                PLUGIN_NAME,
                "chain_type",
                chain_type,
            )
            shared.opts.set(
                PLUGIN_NAME,
                "chunk_limit",
                chunk_limit,
            )
            shared.opts.set(
                PLUGIN_NAME,
                "file_size_limit",
                file_size_limit,
            )
            shared.opts.set(
                PLUGIN_NAME,
                "sentence_size",
                sentence_size,
            )
            shared.opts.set(
                PLUGIN_NAME,
                "using_zh_title_enhance",
                using_zh_title_enhance,
            )
            shared.opts.set(
                PLUGIN_NAME,
                "prompt_template",
                prompt_template,
            )

            return gr.update()

        response = plugin_config_save_btn.click(
            fn=save_plugin_config,
            inputs=[
                chain_type,
                chunk_limit,
                file_size_limit,
                sentence_size,
                using_zh_title_enhance,
                prompt_template,
            ],
            outputs=None,
        )

        return response
