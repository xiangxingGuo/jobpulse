from __future__ import annotations

import os
from typing import Any, Dict

import httpx

from .openai_compat_providers import PROVIDERS, ProviderName

class OpenAICompatClient:
    def __init__(self, provider: ProviderName, timeout: float = 600.0) -> None:
        cfg = PROVIDERS[provider]
        api_key = os.environ.get(cfg.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing {cfg.api_key_env}")

        self.provider = provider
        self.base_url = os.environ.get(f"{provider.upper()}_BASE_URL", cfg.base_url).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                # include response body for debugging (OpenAI returns JSON error)
                body = r.text
                raise RuntimeError(f"HTTP {r.status_code} from {url}: {body}") from e
            return r.json()