"""Shared environment configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required setting: {name}")
    return value


def optional(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def boolean(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}
