# coding=utf-8

import os
import logging

from core import model
from core import plugin
from core import shared
from core.const import *


def initialize():
    if "GRADIO_ANALYTICS_ENABLED" not in os.environ:
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    # log
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    # )
    root = logging.getLogger("root")
    root.setLevel(logging.DEBUG)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("revChatGPT").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.debug("logging init done")

    # config
    shared.opts.load(CONFIG_FILE)

    # model
    model.init_models()
    model.reload_model(shared.opts.get(SYSTEM_CONFIG, DEFAULT_LLM_MODEL))

    # plugin
    plugin.init_plugins()
