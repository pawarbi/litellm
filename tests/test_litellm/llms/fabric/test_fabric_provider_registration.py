"""
Tests for Microsoft Fabric provider registration in LiteLLM.

Verifies that the Fabric provider is properly registered across all
the necessary LiteLLM registration points.

Note: Model list tests that depend on model_prices_and_context_window.json
will only pass after the JSON changes are merged upstream, since LiteLLM
loads the model cost map from a remote URL by default.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.constants import LITELLM_CHAT_PROVIDERS
from litellm.types.utils import LlmProviders, LlmProvidersSet


class TestProviderRegistration:
    def test_fabric_in_llm_providers_enum(self):
        assert hasattr(LlmProviders, "FABRIC")
        assert LlmProviders.FABRIC.value == "fabric"

    def test_fabric_in_llm_providers_set(self):
        assert "fabric" in LlmProvidersSet

    def test_fabric_in_chat_providers_list(self):
        assert "fabric" in LITELLM_CHAT_PROVIDERS

    def test_fabric_in_models_by_provider(self):
        assert "fabric" in litellm.models_by_provider

    def test_get_llm_provider_detects_fabric_prefix(self):
        model, custom_llm_provider, dynamic_api_key, api_base = (
            litellm.get_llm_provider("fabric/gpt-4.1")
        )
        assert custom_llm_provider == "fabric"
        assert model == "gpt-4.1"

    def test_get_llm_provider_detects_fabric_reasoning_model(self):
        model, custom_llm_provider, dynamic_api_key, api_base = (
            litellm.get_llm_provider("fabric/gpt-5")
        )
        assert custom_llm_provider == "fabric"
        assert model == "gpt-5"

    def test_get_llm_provider_detects_fabric_embedding_model(self):
        model, custom_llm_provider, dynamic_api_key, api_base = (
            litellm.get_llm_provider("fabric/text-embedding-ada-002")
        )
        assert custom_llm_provider == "fabric"
        assert model == "text-embedding-ada-002"
