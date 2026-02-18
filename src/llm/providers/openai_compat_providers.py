from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

ProviderName = Literal["openai", "nvidia"]

@dataclass(frozen=True)
class ProviderConfig:
    name: ProviderName
    base_url: str          # e.g. https://api.openai.com or https://integrate.api.nvidia.com/v1
    api_key_env: str       # env var name
    default_model: str

PROVIDERS: Dict[ProviderName, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    ),
    "nvidia": ProviderConfig(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        default_model="moonshotai/kimi-k2.5",
    ),
}
