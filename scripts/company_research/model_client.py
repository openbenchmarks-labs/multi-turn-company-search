"""OpenAI client for the benchmark runner."""

from __future__ import annotations

import os
from typing import Any, Callable


OPENAI_KEY_ENV = "OPENAI_API_KEY"
OPENAI_REQUIRED_ENV = (OPENAI_KEY_ENV,)
MODEL_TRANSPORT = "openai-responses-api"


def build_openai_client(
    client_class: Callable[..., Any] | None = None,
    *,
    timeout: int = 240,
    max_retries: int = 1,
) -> Any:
    """Build the standard OpenAI client used by the research agent."""
    key = os.environ.get(OPENAI_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"missing OpenAI credential: {OPENAI_KEY_ENV}")
    if client_class is None:
        from openai import OpenAI

        client_class = OpenAI
    return client_class(
        api_key=key,
        timeout=timeout,
        max_retries=max_retries,
    )
