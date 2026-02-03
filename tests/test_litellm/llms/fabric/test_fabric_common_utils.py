"""
Tests for Microsoft Fabric provider common utilities.

All tests mock the Fabric SDK since it is only available inside
Fabric notebook environments.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.fabric.common_utils import (
    FABRIC_DEFAULT_API_VERSION,
    FABRIC_REASONING_API_VERSION,
    FabricError,
    build_fabric_url,
    get_fabric_api_version,
    get_fabric_auth_header,
    get_fabric_endpoint,
    is_fabric_environment,
    is_fabric_reasoning_model,
)


class TestIsFabricEnvironment:
    def test_returns_false_outside_fabric(self):
        result = is_fabric_environment()
        assert result is False

    @patch.dict(
        "sys.modules",
        {
            "synapse": MagicMock(),
            "synapse.ml": MagicMock(),
            "synapse.ml.fabric": MagicMock(),
            "synapse.ml.fabric.token_utils": MagicMock(),
        },
    )
    def test_returns_true_when_fabric_sdk_available(self):
        result = is_fabric_environment()
        assert result is True


class TestGetFabricAuthHeader:
    def test_raises_error_outside_fabric(self):
        with pytest.raises(FabricError) as exc_info:
            get_fabric_auth_header()
        assert exc_info.value.status_code == 401
        assert "Fabric" in exc_info.value.message

    def test_returns_auth_header(self):
        mock_token_utils_class = MagicMock()
        mock_token_utils_class.return_value.get_openai_auth_header.return_value = (
            "MwcToken eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
        )

        # Create mock module hierarchy
        mock_synapse = MagicMock()
        mock_synapse.ml.fabric.token_utils.TokenUtils = mock_token_utils_class

        with patch.dict(
            "sys.modules",
            {
                "synapse": mock_synapse,
                "synapse.ml": mock_synapse.ml,
                "synapse.ml.fabric": mock_synapse.ml.fabric,
                "synapse.ml.fabric.token_utils": mock_synapse.ml.fabric.token_utils,
            },
        ):
            result = get_fabric_auth_header()
        assert result == "MwcToken eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"


class TestGetFabricEndpoint:
    def test_uses_explicit_api_base(self):
        result = get_fabric_endpoint(api_base="https://my-endpoint.com/workspace/")
        assert result == "https://my-endpoint.com/workspace/"

    def test_adds_trailing_slash(self):
        result = get_fabric_endpoint(api_base="https://my-endpoint.com/workspace")
        assert result == "https://my-endpoint.com/workspace/"

    def test_uses_env_var(self):
        with patch.dict(
            os.environ, {"FABRIC_API_BASE": "https://env-endpoint.com/ws/"}
        ):
            result = get_fabric_endpoint()
        assert result == "https://env-endpoint.com/ws/"

    def test_raises_error_when_no_endpoint_and_outside_fabric(self):
        with patch.dict(os.environ, {}, clear=True):
            # Make sure FABRIC_API_BASE is not set
            os.environ.pop("FABRIC_API_BASE", None)
            with pytest.raises(FabricError) as exc_info:
                get_fabric_endpoint()
            assert exc_info.value.status_code == 401

    def test_auto_discovers_from_fabric_sdk(self):
        mock_config = MagicMock()
        mock_config.fabric_env_config.ml_workload_endpoint = (
            "https://auto-discovered.com/ws/"
        )
        mock_discovery = MagicMock()
        mock_discovery.get_fabric_env_config.return_value = mock_config

        with patch.dict(
            "sys.modules",
            {
                "synapse": MagicMock(),
                "synapse.ml": MagicMock(),
                "synapse.ml.fabric": MagicMock(),
                "synapse.ml.fabric.service_discovery": mock_discovery,
            },
        ):
            result = get_fabric_endpoint()
        assert result == "https://auto-discovered.com/ws/"


class TestIsFabricReasoningModel:
    def test_gpt5_is_reasoning(self):
        assert is_fabric_reasoning_model("gpt-5") is True

    def test_gpt5_mini_is_reasoning(self):
        assert is_fabric_reasoning_model("gpt-5-mini") is True

    def test_gpt41_is_not_reasoning(self):
        assert is_fabric_reasoning_model("gpt-4.1") is False

    def test_gpt41_mini_is_not_reasoning(self):
        assert is_fabric_reasoning_model("gpt-4.1-mini") is False

    def test_embedding_is_not_reasoning(self):
        assert is_fabric_reasoning_model("text-embedding-ada-002") is False


class TestGetFabricApiVersion:
    def test_standard_model_gets_default_version(self):
        result = get_fabric_api_version("gpt-4.1")
        assert result == FABRIC_DEFAULT_API_VERSION

    def test_reasoning_model_gets_reasoning_version(self):
        result = get_fabric_api_version("gpt-5")
        assert result == FABRIC_REASONING_API_VERSION

    def test_explicit_override(self):
        result = get_fabric_api_version("gpt-4.1", api_version="2099-01-01")
        assert result == "2099-01-01"

    def test_explicit_override_takes_precedence_over_reasoning(self):
        result = get_fabric_api_version("gpt-5", api_version="2099-01-01")
        assert result == "2099-01-01"


class TestBuildFabricUrl:
    def test_chat_completions_url(self):
        url = build_fabric_url(
            endpoint="https://example.com/ws/",
            model="gpt-4.1",
            api_version="2024-02-15-preview",
            endpoint_type="chat/completions",
        )
        assert url == (
            "https://example.com/ws/"
            "cognitive/openai/openai/deployments/"
            "gpt-4.1/chat/completions?api-version=2024-02-15-preview"
        )

    def test_embeddings_url(self):
        url = build_fabric_url(
            endpoint="https://example.com/ws/",
            model="text-embedding-ada-002",
            api_version="2024-02-15-preview",
            endpoint_type="embeddings",
        )
        assert url == (
            "https://example.com/ws/"
            "cognitive/openai/openai/deployments/"
            "text-embedding-ada-002/embeddings?api-version=2024-02-15-preview"
        )

    def test_reasoning_model_url(self):
        url = build_fabric_url(
            endpoint="https://example.com/ws/",
            model="gpt-5",
            api_version="2025-04-01-preview",
            endpoint_type="chat/completions",
        )
        assert url == (
            "https://example.com/ws/"
            "cognitive/openai/openai/deployments/"
            "gpt-5/chat/completions?api-version=2025-04-01-preview"
        )
