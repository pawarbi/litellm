"""
Tests for Microsoft Fabric provider chat completion handler.

All tests mock both the Fabric SDK and HTTP responses since the Fabric SDK
is only available inside Fabric notebook environments.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.fabric.chat.handler import FabricChatCompletion
from litellm.llms.fabric.common_utils import FabricError
from litellm.types.utils import ModelResponse

MOCK_AUTH_HEADER = "MwcToken eyJhbGciOiJSUzI1NiJ9.test"
MOCK_ENDPOINT = "https://test.pbidedicated.windows.net/webapi/ws/"

MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4.1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Fabric."},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
}


def _mock_logging_obj():
    obj = MagicMock()
    obj.pre_call = MagicMock()
    obj.post_call = MagicMock()
    return obj


def _mock_httpx_response(data, status_code=200):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = data
    mock_response.text = json.dumps(data)
    mock_response.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="error",
            request=httpx.Request("POST", "https://test.com"),
            response=httpx.Response(status_code),
        )
    return mock_response


def _mock_http_client(response_data, status_code=200):
    """Create a mock HTTPHandler that passes isinstance checks."""
    mock_client = MagicMock(spec=HTTPHandler)
    mock_client.post.return_value = _mock_httpx_response(response_data, status_code)
    return mock_client


class TestFabricChatCompletionSync:
    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_auth_header",
        return_value=MOCK_AUTH_HEADER,
    )
    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_endpoint",
        return_value=MOCK_ENDPOINT,
    )
    def test_completion_sends_correct_url(self, mock_endpoint, mock_auth):
        handler = FabricChatCompletion()
        mock_client = _mock_http_client(MOCK_CHAT_RESPONSE)

        handler.completion(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Hello"}],
            model_response=ModelResponse(),
            optional_params={},
            litellm_params={},
            logging_obj=_mock_logging_obj(),
            client=mock_client,
        )

        call_args = mock_client.post.call_args
        url = call_args.kwargs.get("url", "")
        assert "cognitive/openai/openai/deployments/gpt-4.1/chat/completions" in url
        assert "api-version=2024-02-15-preview" in url

    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_auth_header",
        return_value=MOCK_AUTH_HEADER,
    )
    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_endpoint",
        return_value=MOCK_ENDPOINT,
    )
    def test_completion_sends_mwctoken_auth(self, mock_endpoint, mock_auth):
        handler = FabricChatCompletion()
        mock_client = _mock_http_client(MOCK_CHAT_RESPONSE)

        handler.completion(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Hello"}],
            model_response=ModelResponse(),
            optional_params={},
            litellm_params={},
            logging_obj=_mock_logging_obj(),
            client=mock_client,
        )

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers")
        assert headers is not None
        assert headers["Authorization"] == MOCK_AUTH_HEADER
        assert headers["Authorization"].startswith("MwcToken ")

    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_auth_header",
        return_value=MOCK_AUTH_HEADER,
    )
    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_endpoint",
        return_value=MOCK_ENDPOINT,
    )
    def test_completion_returns_model_response(self, mock_endpoint, mock_auth):
        handler = FabricChatCompletion()
        mock_client = _mock_http_client(MOCK_CHAT_RESPONSE)

        result = handler.completion(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Hello"}],
            model_response=ModelResponse(),
            optional_params={},
            litellm_params={},
            logging_obj=_mock_logging_obj(),
            client=mock_client,
        )

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "Hello from Fabric."
        assert result.usage.total_tokens == 15

    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_auth_header",
        return_value=MOCK_AUTH_HEADER,
    )
    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_endpoint",
        return_value=MOCK_ENDPOINT,
    )
    def test_reasoning_model_uses_correct_api_version(self, mock_endpoint, mock_auth):
        handler = FabricChatCompletion()
        mock_client = _mock_http_client(MOCK_CHAT_RESPONSE)

        handler.completion(
            model="gpt-5",
            messages=[{"role": "user", "content": "Hello"}],
            model_response=ModelResponse(),
            optional_params={},
            litellm_params={},
            logging_obj=_mock_logging_obj(),
            client=mock_client,
        )

        call_args = mock_client.post.call_args
        url = call_args.kwargs.get("url", "")
        assert "api-version=2025-04-01-preview" in url

    def test_completion_raises_on_missing_fabric_sdk(self):
        handler = FabricChatCompletion()
        with pytest.raises(FabricError) as exc_info:
            handler.completion(
                model="gpt-4.1",
                messages=[{"role": "user", "content": "Hello"}],
                model_response=ModelResponse(),
                optional_params={},
                litellm_params={},
                logging_obj=_mock_logging_obj(),
            )
        assert exc_info.value.status_code == 401

    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_auth_header",
        return_value=MOCK_AUTH_HEADER,
    )
    @patch(
        "litellm.llms.fabric.chat.handler.get_fabric_endpoint",
        return_value=MOCK_ENDPOINT,
    )
    def test_completion_sends_messages_in_payload(self, mock_endpoint, mock_auth):
        handler = FabricChatCompletion()
        mock_client = _mock_http_client(MOCK_CHAT_RESPONSE)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        handler.completion(
            model="gpt-4.1",
            messages=messages,
            model_response=ModelResponse(),
            optional_params={},
            litellm_params={},
            logging_obj=_mock_logging_obj(),
            client=mock_client,
        )

        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args.kwargs.get("data", "{}"))
        assert sent_data["messages"] == messages
