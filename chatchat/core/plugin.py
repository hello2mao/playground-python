# coding=utf-8

import logging, os
import gradio as gr
from typing import List
from retry import retry

from modules.plugins import Summarization
from modules.plugins import SQL
from modules.plugins import *
from modules.plugins import BasePlugin
from modules.utils.log import record_log
from core import shared
from core.const import *


@record_log
def init_plugins():
    for plugin_name in shared.opts.get(SYSTEM_CONFIG, PLUGINS):
        class_object: BasePlugin = globals().get(plugin_name) or locals().get(
            plugin_name
        )
        if class_object is None:
            logging.error(f"init_plugins failed: class_object is None")
            os._exit(-1)
        shared.plugins[plugin_name] = class_object()


def plugin_change(plugin_choice):
    plugin_choice = shared.opts.from_display_name(plugin_choice)
    results = []
    for plugin_name in shared.opts.get(SYSTEM_CONFIG, PLUGINS):
        results.append(gr.update(visible=plugin_name == plugin_choice))
    shared.cur_plugin_name = plugin_choice
    return results


def plugin_config_change(plugin_choice):
    plugin_choice = shared.opts.from_display_name(plugin_choice)
    results = []
    for plugin_name in shared.opts.get(SYSTEM_CONFIG, PLUGINS):
        results.append(gr.update(visible=plugin_name == plugin_choice))
    return results
