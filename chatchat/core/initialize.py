import os
import logging

from core import config
from core import model


def initialize():
    if "GRADIO_ANALYTICS_ENABLED" not in os.environ:
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    # log
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("revChatGPT").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # config
    conf = config.init()

    # model
    model.init_models()
    model.reload_model(conf["default_llm_model"])
