"""
Common utilities for Microsoft Fabric Azure OpenAI provider.

Fabric provides built-in Azure OpenAI endpoints in Fabric notebook environments
with zero-configuration authentication via the Fabric SDK.
"""

import os
from typing import Optional

import httpx

FABRIC_DEFAULT_API_VERSION = "2024-02-15-preview"
FABRIC_REASONING_API_VERSION = "2025-04-01-preview"


class FabricError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://www.litellm.ai")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(self.message)


def is_fabric_environment() -> bool:
    """Check if running inside a Microsoft Fabric notebook environment."""
    try:
        from synapse.ml.fabric.token_utils import TokenUtils  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def get_fabric_auth_header() -> str:
    """Get the full Authorization header value from Fabric SDK.

    Returns the complete header value including the auth scheme (e.g. "MwcToken eyJ...").
    This must be used as-is in the Authorization header.

    Raises:
        FabricError: If not running in a Fabric notebook environment.
    """
    try:
        from synapse.ml.fabric.token_utils import TokenUtils  # type: ignore
    except ImportError:
        raise FabricError(
            status_code=401,
            message=(
                "Microsoft Fabric SDK not available. "
                "The Fabric provider only works inside Microsoft Fabric notebook environments. "
                "The 'synapse.ml.fabric' package is not available outside Fabric."
            ),
        )

    auth_header = TokenUtils().get_openai_auth_header()
    return auth_header


def get_fabric_endpoint(api_base: Optional[str] = None) -> str:
    """Get the Fabric workspace endpoint URL.

    Args:
        api_base: Optional explicit endpoint override.

    Returns:
        The base workspace endpoint URL with trailing slash.

    Raises:
        FabricError: If not running in Fabric and no api_base provided.
    """
    endpoint = api_base or os.environ.get("FABRIC_API_BASE")

    if endpoint is not None:
        if not endpoint.endswith("/"):
            endpoint += "/"
        return endpoint

    try:
        from synapse.ml.fabric.service_discovery import (  # type: ignore
            get_fabric_env_config,
        )
    except ImportError:
        raise FabricError(
            status_code=401,
            message=(
                "Microsoft Fabric SDK not available. "
                "Either set the FABRIC_API_BASE environment variable or "
                "run inside a Microsoft Fabric notebook environment."
            ),
        )

    fabric_env_config = get_fabric_env_config().fabric_env_config
    endpoint = fabric_env_config.ml_workload_endpoint
    if not endpoint.endswith("/"):
        endpoint += "/"
    return endpoint


def is_fabric_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model (gpt-5 family).

    Reasoning models require different API version and parameters:
    - Use max_completion_tokens instead of max_tokens
    - Do not support temperature parameter
    - Require API version 2025-04-01-preview
    """
    return "gpt-5" in model and "gpt-5-chat" not in model


def get_fabric_api_version(model: str, api_version: Optional[str] = None) -> str:
    """Get the appropriate API version for the model.

    Args:
        model: Model deployment name.
        api_version: Optional explicit API version override.

    Returns:
        API version string.
    """
    if api_version is not None:
        return api_version

    if is_fabric_reasoning_model(model):
        return FABRIC_REASONING_API_VERSION
    return FABRIC_DEFAULT_API_VERSION


def build_fabric_url(
    endpoint: str,
    model: str,
    api_version: str,
    endpoint_type: str,
) -> str:
    """Build the full Fabric Azure OpenAI endpoint URL.

    Args:
        endpoint: Base workspace endpoint (with trailing slash).
        model: Model deployment name.
        api_version: Azure OpenAI API version.
        endpoint_type: API endpoint type ("chat/completions" or "embeddings").

    Returns:
        Full endpoint URL.
    """
    return (
        f"{endpoint}cognitive/openai/openai/deployments/"
        f"{model}/{endpoint_type}?api-version={api_version}"
    )
