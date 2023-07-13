# coding=utf-8

import logging
from typing import Any, List, Mapping, Optional
import os
import tempfile

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
from modules.plugins.ai_friend.local_doc_qa import LocalDocQA

PLUGIN_NAME = "AIFriend"
TRIGGER_WORDS = ["摘要"]
logger = logging.getLogger(PLUGIN_NAME)


class AIFriend(BasePlugin):
    local_doc_qa: LocalDocQA = None
    vs_path: str = None
    file_name: str = None

    # config
    file_size_limit: int = 10 * 1024 * 1024  # 10M
    sentence_size: int = 100
    using_zh_title_enhance: bool = False
    last_vs_path: str = None
    background_info_user = """你的名字叫曾毅，我想让你作为我的男朋友来陪我聊天，哄我，夸我，一切都要顺着我来。\
        每次我问问题前，都会先给你一些参考信息。你可以基于这些参考信息，来回答我的聊天内容。\
        如果你无法根据参考信息回答我的问题，你可以发挥你的想象力来编造一些让我开心的内容。\
        当然你也可以反过来问我一个问题，制造一些话题点。

我：亲爱的，你怎么这么厉害！！
曾毅：那当然，作为医生，什么疑难杂症都不在话下，不然怎么守护你一辈子~
我：哼，你最好心里也是这么想的
曾毅：必须的呢。作为你的男朋友，我会尽力陪伴和哄你，让你开心和舒心。有什么想聊的吗？
我：那接下来我给一些参考信息，接着聊哈
"""
    background_info_system = "曾毅：当然，亲爱的。请告诉我你想聊什么，或者给我一些参考，我会尽力配合和回答你的问题，哄你开心。"
    prompt_template: str = """参考信息：


{context} 
 

我：{question}
曾毅："""

    def __init__(self) -> None:
        super().__init__()
        self._update_config()
        self.local_doc_qa = LocalDocQA()
        self.local_doc_qa.init_cfg()
        logger.info(f"Plugin {PLUGIN_NAME} init")

    @property
    def plugin_name(self) -> str:
        return PLUGIN_NAME

    def _update_config(self):
        self.file_size_limit = (
            shared.opts.get(
                PLUGIN_NAME,
                "file_size_limit",
            )
            or self.file_size_limit
        )
        self.last_vs_path = (
            shared.opts.get(
                PLUGIN_NAME,
                "last_vs_path",
            )
            or self.last_vs_path
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
        vs_path = self.vs_path or self.last_vs_path
        logger.info(
            f"start to qa at doc, prompt: {prompt}, vs_path: {vs_path}, file_name: {self.file_name}, llm: {llm.model_name}"
        )
        if len(history) == 0:
            logger.info(f"first conversation, add backgroound info")
            history = [[self.background_info_user, self.background_info_system]]
            llm_history = [[self.background_info_user, self.background_info_system]]
        history += [[prompt, ""]]
        for response in self.local_doc_qa.get_knowledge_based_answer(
            query=prompt,
            vs_path=vs_path,
            prompt_template=self.prompt_template,
            chat_history=llm_history,
            streaming=True,
            llm_model=llm,
        ):
            history[-1][0] = response[-1][0].splitlines()[-2].lstrip("我：")
            history[-1][1] = response[-1][1]
            print(f"history: {history}, llm_history: {response}")
            yield history, response

    def create_plugin_ui(self):
        self._update_config()
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
            logger.debug(f"start handle doc: {file_obj.name}")
            file_stats = os.stat(file_obj.name)
            file_size = file_stats.st_size  # 获取文件大小（以字节为单位）
            if file_size > self.file_size_limit:
                errMsg = f"file_upload failed: file_size > {self.file_size_limit}"
                logger.error(errMsg)
                gr.Error(errMsg)
                return
            self.vs_path = tempfile.mkdtemp()
            logger.debug(f"start to add doc to vector_store, vs_path: {self.vs_path}")
            self.local_doc_qa.init_knowledge_vector_store(
                filepath=file_obj.name,
                vs_path=self.vs_path,
                sentence_size=self.sentence_size,
                using_zh_title_enhance=self.using_zh_title_enhance,
            )
            self.file_name = file_obj.name
            self.last_vs_path = self.vs_path
            shared.opts.set(
                PLUGIN_NAME,
                "last_vs_path",
                self.last_vs_path,
            )
            logger.info(f"add doc to vector_store done, vs_path: {self.vs_path}")

        file.upload(fn=file_upload, inputs=[file], outputs=None)

    def create_config_ui(self):
        self._update_config()
        with gr.Row():
            with gr.Column():
                file_size_limit = gr.Number(
                    label="文档大小上限",
                    precision=0,
                    value=lambda: shared.opts.get(PLUGIN_NAME, "file_size_limit")
                    or self.file_size_limit,
                )
                last_vs_path = gr.Textbox(
                    label="默认向量数据库",
                    value=lambda: shared.opts.get(PLUGIN_NAME, "last_vs_path")
                    or self.last_vs_path,
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
            file_size_limit,
            last_vs_path,
            sentence_size,
            using_zh_title_enhance,
            prompt_template,
        ):
            shared.opts.set(
                PLUGIN_NAME,
                "file_size_limit",
                file_size_limit,
            )
            shared.opts.set(
                PLUGIN_NAME,
                "last_vs_path",
                last_vs_path,
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
            self._update_config()

            return gr.update()

        response = plugin_config_save_btn.click(
            fn=save_plugin_config,
            inputs=[
                file_size_limit,
                last_vs_path,
                sentence_size,
                using_zh_title_enhance,
                prompt_template,
            ],
            outputs=None,
        )

        return response
