# coding=utf-8

from typing import Optional, List, Dict
import gradio as gr
import torch
import logging
import time
from retry import retry

from ..base import BaseLLM
from modules.utils import gpu
from core import shared
from core import model

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_NAME = "Baichuan"
logger = logging.getLogger(MODEL_NAME)

BAICHUAN_MODELS = [
    "baichuan-inc/Baichuan-13B-Chat",
    "baichuan-inc/Baichuan-13B-Base",
    "baichuan-inc/Baichuan-7B",
]


class Baichuan(BaseLLM):
    # model
    tokenizer: object = None
    model: object = None
    model_config: object = None

    # config
    pretrained_model_name_or_path: str = BAICHUAN_MODELS[0]
    bf16: bool = False
    device: str = "cuda"  # TODO

    def __init__(self):
        super().__init__()
        logger.info(f"Model {MODEL_NAME} init")

    @property
    def model_name(self) -> str:
        return MODEL_NAME

    def _load_model_config(self):
        model_config = GenerationConfig.from_pretrained(
            self.pretrained_model_name_or_path, trust_remote_code=True
        )

        return model_config

    def _load_model(self):
        config = shared.opts.get(MODEL_NAME, None)
        logger.info(f"model config: {config}")
        self.pretrained_model_name_or_path = config.get(
            "pretrained_model_name_or_path", BAICHUAN_MODELS[0]
        )
        self.bf16 = config.get("bf16", False)

        logger.info(f"Loading {self.pretrained_model_name_or_path}...")

        t0 = time.time()
        # load model
        model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float16,
            device_map="auto",
            generation_config=self.model_config,
            trust_remote_code=True,
        )

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

        logger.info(f"Loaded the model in {(time.time() - t0):.2f} seconds.")
        return model, tokenizer

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model = self.tokenizer = None
        gpu.clear_torch_cache(self.device)

    @retry(tries=3, delay=1)
    def reload_model(self):
        self.unload_model()
        self.model_config = self._load_model_config()
        self.model, self.tokenizer = self._load_model()
        self.model = self.model.eval()

    def generateAnswer(
        self, prompt: str, history: List[List[str]] = [], streaming: bool = False
    ) -> List[List[str]]:
        messages = []
        for index, message in enumerate(history):
            if index % 2 == 0:
                messages.append({"role": "user", "content": message})
            else:
                messages.append({"role": "assistant", "content": message})
        messages.append({"role": "user", "content": prompt})
        if streaming:
            for stream_resp in self.model.chat(self.tokenizer, messages, stream=True):
                history[-1] = [prompt, stream_resp]
                yield history
        else:
            for stream_resp in self.model.chat(self.tokenizer, messages, stream=True):
                history[-1] = [prompt, stream_resp]
            gpu.clear_torch_cache()
            yield history

    def create_config_ui(self):
        pretrained_model_name_or_path = gr.Radio(
            label="子模型选择",
            choices=BAICHUAN_MODELS,
            value=lambda: shared.opts.get(MODEL_NAME, "pretrained_model_name_or_path"),
        )
        model_config_save_btn = gr.Button(
            "保存并加载",
            elem_id="model_config_save",
            variant="primary",
        )

        def save_model_config(
            pretrained_model_name_or_path, progress=gr.Progress(track_tqdm=True)
        ):
            shared.opts.set(
                MODEL_NAME,
                "pretrained_model_name_or_path",
                pretrained_model_name_or_path,
            )
            model.reload_model(MODEL_NAME)
            return gr.update()

        response = model_config_save_btn.click(
            fn=save_model_config,
            inputs=[pretrained_model_name_or_path],
            outputs=[pretrained_model_name_or_path],
        )

        return response
