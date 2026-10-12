"""Runtime configuration (read when Settings() is created)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    api_keys: list[str] = field(default_factory=lambda: [k for k in os.getenv("API_KEYS", "dev-key").split(",") if k])


def get_settings() -> Settings:
    return Settings()
