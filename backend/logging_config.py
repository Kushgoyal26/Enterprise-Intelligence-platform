"""
Centralized logging setup. Import `get_logger` wherever you need structured,
timestamped logs instead of scattered print() statements.

In production, logs like this are what you'd ship to a monitoring tool
(e.g. Grafana/Loki, Datadog, CloudWatch) instead of reading a terminal.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on reload
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
