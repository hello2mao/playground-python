# coding=utf-8

import gc
import torch


def clear_torch_cache(cuda_device: str = "cuda"):
    gc.collect()
    if cuda_device is not None:
        if torch.has_mps:
            try:
                from torch.mps import empty_cache

                empty_cache()
            except Exception as e:
                print(f"clear_torch_cache err: {e}")
        elif torch.has_cuda:
            with torch.cuda.device(cuda_device + ":0"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
