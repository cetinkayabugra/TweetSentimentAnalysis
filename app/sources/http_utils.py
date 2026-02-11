from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass(slots=True)
class SimpleResponse:
    status_code: int
    _payload: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self._payload


def get_json(url: str, params: dict[str, Any], timeout: int) -> SimpleResponse:
    query = urlencode(params)
    full_url = f"{url}?{query}"
    with urlopen(full_url, timeout=timeout) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
        return SimpleResponse(status_code=getattr(resp, "status", 200), _payload=json.loads(body))


def get_client():
    try:
        import requests  # type: ignore

        return requests
    except ImportError:
        class _Fallback:
            @staticmethod
            def get(url: str, params: dict[str, Any], timeout: int):
                return get_json(url, params=params, timeout=timeout)

        return _Fallback()
