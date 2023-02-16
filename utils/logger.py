import logging
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    if isinstance(log_file, Path):
        log_file = str(log_file)

    logger = logging.getLogger()
    datefmt = "%Y-%m-%d %H:%M:%S"
    fmt = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)s] -   %(message)s"
    log_format = logging.Formatter(fmt=fmt, datefmt=datefmt)

    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = TimedRotatingFileHandler(filename=log_file, when="D", backupCount=7)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
