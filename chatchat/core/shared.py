from typing import Dict

from core.models.model import Model

import gradio as gr

app: gr.Blocks = None
conf: Dict = None
model: Model = None
