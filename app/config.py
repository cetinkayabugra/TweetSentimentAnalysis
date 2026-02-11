from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(slots=True)
class AppConfig:
    """Application runtime configuration."""

    cache_dir: Path = Path("cache")
    cache_ttl_seconds: int = 30 * 60
    request_timeout_seconds: int = 20
    max_retries: int = 4
    backoff_base_seconds: float = 1.0
    alpha_vantage_api_key: str | None = field(default_factory=lambda: os.getenv("ALPHAVANTAGE_API_KEY"))
    newsapi_key: str | None = field(default_factory=lambda: os.getenv("NEWSAPI_KEY"))
    source_credibility_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "reuters": 1.2,
            "bloomberg": 1.2,
            "the wall street journal": 1.15,
            "financial times": 1.15,
            "cnbc": 1.05,
        }
    )


DEFAULT_CONFIG = AppConfig()
