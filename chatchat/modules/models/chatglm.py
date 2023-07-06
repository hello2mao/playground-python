import torch
import gradio as gr
from pathlib import Path
from typing import Optional, List, Dict
import time
import logging


from .base import BaseModel
from modules.utils import gpu

from transformers import AutoTokenizer, AutoModel, AutoConfig

logger = logging.getLogger("ChatGLM")

CHATGLM_MODELS = [
    "THUDM/chatglm2-6b",
    "THUDM/chatglm-6b",
    "THUDM/chatglm-6b-int8",
    "THUDM/chatglm-6b-int4",
    "THUDM/chatglm-6b-int4-qe",
]


class ChatGLM(BaseModel):
    # model
    tokenizer: object = None
    model: object = None
    model_config: object = None

    # config
    model_name: str = CHATGLM_MODELS[0]
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history_len: int = 10
    bf16: bool = False
    lora: str = None  # TODO
    device: str = "cuda"  # TODO
    device_map: Optional[Dict[str, int]] = None  # TODO

    def __init__(self, config: dict = None):
        super().__init__()
        self.model_name = config.get("model_name", CHATGLM_MODELS[0])
        self.max_token = config.get("max_token", 10000)
        self.temperature = config.get("temperature", 0.01)
        self.top_p = config.get("top_p", 0.9)
        self.history_len = config.get("history_len", 10)
        self.bf16 = config.get("bf16", False)

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _load_model(self):
        logger.info(f"Loading {self.model_name}...")

        t0 = time.time()

        # load model
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2 and self.device_map is None:
            model = (
                AutoModel.from_pretrained(
                    self.model_name,
                    config=self.model_config,
                    torch_dtype=torch.bfloat16 if self.bf16 else torch.float16,
                    trust_remote_code=True,
                )
                .half()
                .cuda()
            )
        else:
            from accelerate import dispatch_model

            model = AutoModel.from_pretrained(
                self.model_name,
                config=self.model_config,
                torch_dtype=torch.bfloat16 if self.bf16 else torch.float16,
                trust_remote_code=True,
            ).half()
            if self.device_map is not None:
                self.device_map = self._auto_configure_device_map(num_gpus)

            model = dispatch_model(model, device_map=self.device_map)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        logger.info(f"Loaded the model in {(time.time() - t0):.2f} seconds.")
        return model, tokenizer

    def _auto_configure_device_map(self, num_gpus: int) -> Dict[str, int]:
        # transformer.word_embeddings 占用1层
        # transformer.final_layernorm 和 lm_head 占用1层
        # transformer.layers 占用 28 层
        # 总共30层分配到num_gpus张卡上
        num_trans_layers = 28
        per_gpu_layers = 30 / num_gpus

        # bugfix: PEFT加载lora模型出现的层命名不同
        if self.lora:
            layer_prefix = "base_model.model.transformer"
        else:
            layer_prefix = "transformer"

        # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
        # windows下 model.device 会被设置成 transformer.word_embeddings.device
        # linux下 model.device 会被设置成 lm_head.device
        # 在调用chat或者stream_chat时,input_ids会被放到model.device上
        # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
        # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
        device_map = {
            f"{layer_prefix}.word_embeddings": 0,
            f"{layer_prefix}.final_layernorm": 0,
            "lm_head": 0,
            f"base_model.model.lm_head": 0,
        }

        used = 2
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f"{layer_prefix}.layers.{i}"] = gpu_target
            used += 1

        return device_map

    def _load_model_config(self):
        model_config = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model_config

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model = self.tokenizer = None
        gpu.clear_torch_cache(self.device)

    def reload_model(self):
        self.unload_model()
        self.model_config = self._load_model_config()
        self.model, self.tokenizer = self._load_model()
        self.model = self.model.eval()

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        logger.debug(f"__call: {prompt}")
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            max_length=self.max_token,
            temperature=self.temperature,
        )
        logger.debug(f"response: {response}")
        return response

    def generateAnswer(
        self, prompt: str, history: List[List[str]] = [], streaming: bool = False
    ):
        if streaming:
            history += [[]]
            for stream_resp, _ in self.model.stream_chat(
                self.tokenizer,
                prompt,
                history=history[-self.history_len : -1] if self.history_len > 1 else [],
                max_length=self.max_token,
                temperature=self.temperature,
            ):
                history[-1] = [prompt, stream_resp]
                yield history
        else:
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=history[-self.history_len :] if self.history_len > 0 else [],
                max_length=self.max_token,
                temperature=self.temperature,
            )
            gpu.clear_torch_cache()
            history += [[prompt, response]]
            return history

    def create_config_ui(self):
        model_name = gr.Radio(choices=CHATGLM_MODELS, value=self.model_name)
        return [model_name]
