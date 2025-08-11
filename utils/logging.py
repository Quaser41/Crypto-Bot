import logging
import os


def get_logger(name: str = __name__):
    """Configure and return a module-level logger.

    Log level can be controlled via the ``LOG_LEVEL`` environment variable.
    Defaults to ``INFO`` if unspecified or invalid.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)

    return logging.getLogger(name)
