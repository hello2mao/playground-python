import logging
import re
import yaml

from core import shared

CONFIG_FILE = "config.yml"


def init():
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with open(CONFIG_FILE) as f:
        shared.conf = yaml.load(f, Loader=loader)
        logging.info(f"config is: {shared.conf}")

    return shared.conf


def save(conf):
    with open(CONFIG_FILE) as f:
        yaml.dump(conf, f)
