import logging
from typing import Union

def setup_logging(log_level: Union[int, str] = logging.INFO):
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)
    logging.basicConfig(
        format="{asctime} - {levelname} - [{filename} {funcName}() {lineno}] - pid {process} - {message}",
        filename=None, force=True,
        level=log_level, style="{") # logging level can be DEBUG, INFO, WARNING, ERROR, CRITICAL
