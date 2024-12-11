import logging
import sys
from typing import Optional


def setup_logger(
    debug: bool = False, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the symbolic regression package.

    Args:
        debug: If True, sets logging level to DEBUG, otherwise INFO
        log_file: Optional file path to save logs

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("symb_regression")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
