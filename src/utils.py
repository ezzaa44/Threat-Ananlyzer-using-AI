import logging
from pathlib import Path

from src import config


def ensure_project_dirs() -> None:
    dirs = [
        config.DATA_DIR / "processed",
        config.MODEL_PATH.parent,
        config.REPORTS_DIR,
        config.LOG_PATH.parent,
    ]
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_PATH),
            logging.StreamHandler(),
        ],
    )
