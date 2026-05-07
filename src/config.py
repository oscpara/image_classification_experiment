from __future__ import annotations

import configparser
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.ini"


def get_database_url() -> str | None:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return database_url

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(CONFIG_PATH, encoding="utf-8-sig")
    return parser.get("database", "url", fallback=None)
