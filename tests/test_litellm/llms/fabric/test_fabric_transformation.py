"""
Tests for Microsoft Fabric provider parameter transformation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.fabric.chat.transformation import FabricConfig


class TestFabricConfigSupportedParams:
    def test_standard_model_supports_temperature(self):
        config = FabricConfig()
        params = config.get_supported_openai_params("gpt-4.1")
        assert "temperature" in params

    def test_reasoning_model_does_not_support_temperature(self):
        config = FabricConfig()
        params = config.get_supported_openai_params("gpt-5")
        assert "temperature" not in params

    def test_reasoning_model_does_not_support_n(self):
        config = FabricConfig()
        params = config.get_supported_openai_params("gpt-5")
        assert "n" not in params

    def test_reasoning_model_supports_max_completion_tokens(self):
        config = FabricConfig()
        params = config.get_supported_openai_params("gpt-5")
        assert "max_completion_tokens" in params

    def test_standard_model_supports_common_params(self):
        config = FabricConfig()
        params = config.get_supported_openai_params("gpt-4.1")
        assert "messages" in params or "max_tokens" in params
        assert "stream" in params


class TestFabricConfigMapParams:
    def test_standard_model_passes_temperature(self):
        config = FabricConfig()
        result = config.map_openai_params(
            non_default_params={"temperature": 0.7, "max_tokens": 100},
            optional_params={},
            model="gpt-4.1",
            drop_params=False,
            api_version="2024-02-15-preview",
        )
        assert result.get("temperature") == 0.7

    def test_reasoning_model_converts_max_tokens(self):
        config = FabricConfig()
        result = config.map_openai_params(
            non_default_params={"max_tokens": 500},
            optional_params={},
            model="gpt-5",
            drop_params=False,
            api_version="2025-04-01-preview",
        )
        assert "max_tokens" not in result
        assert result.get("max_completion_tokens") == 500

    def test_reasoning_model_strips_temperature(self):
        config = FabricConfig()
        result = config.map_openai_params(
            non_default_params={"temperature": 0.7, "max_tokens": 100},
            optional_params={},
            model="gpt-5",
            drop_params=False,
            api_version="2025-04-01-preview",
        )
        assert "temperature" not in result

    def test_reasoning_model_preserves_max_completion_tokens(self):
        config = FabricConfig()
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 1000},
            optional_params={},
            model="gpt-5",
            drop_params=False,
            api_version="2025-04-01-preview",
        )
        assert result.get("max_completion_tokens") == 1000
