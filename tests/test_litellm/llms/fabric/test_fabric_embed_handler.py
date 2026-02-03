"""
Tests for Microsoft Fabric provider embedding handler.

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
from litellm.llms.fabric.common_utils import FabricError
from litellm.llms.fabric.embed.handler import FabricEmbeddingHandler
from litellm.types.utils import EmbeddingResponse

MOCK_AUTH_HEADER = "MwcToken eyJhbGciOiJSUzI1NiJ9.test"
MOCK_ENDPOINT = "https://test.pbidedicated.windows.net/webapi/ws/"

MOCK_EMBEDDING_RESPONSE = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.1, 0.2, 0.3],
            "index": 0,
        }
    ],
    "model": "text-embedding-ada-002",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
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


class TestFabricEmbeddingSync:
    @patch(
        "litellm.llms.fabric.embed.handler.get_fabric_auth_header",
        return_value=MOCK_AUTH_HEADER,
    )
    @patch(
        "litellm.llms.fabric.embed.handler.get_fabric_endpoint",
        return_value=MOCK_ENDPOINT,
    )
    def test_embedding_sends_correct_url(self, mock_endpoint, mock_auth):
        handler = FabricEmbeddingHandler()
        mock_client = _mock_http_client(MOCK_EMBEDDING_RESPONSE)

        handler.embedding(
            model="text-embedding-ada-002",
            input=["Hello world"],
            model_response=EmbeddingResponse(),
            optional_params={},
            logging_obj=_mock_logging_obj(),
            client=mock_client,
        )

        call_args = mock_client.post.call_args
        url = call_args.kwargs.get("url", "")
        assert (
            "cognitive/openai/openai/deployments/text-embedding-ada-002/embeddings"
            in url
        )

    @patch(
        "litellm.llms.fabric.embed.handler.get_fabric_auth_header",
        return_value=MOCK_AUTH_HEADER,
    )
    @patch(
        "litellm.llms.fabric.embed.handler.get_fabric_endpoint",
        return_value=MOCK_ENDPOINT,
    )
    def test_embedding_returns_embedding_response(self, mock_endpoint, mock_auth):
        handler = FabricEmbeddingHandler()
        mock_client = _mock_http_client(MOCK_EMBEDDING_RESPONSE)

        result = handler.embedding(
            model="text-embedding-ada-002",
            input=["Hello world"],
            model_response=EmbeddingResponse(),
            optional_params={},
            logging_obj=_mock_logging_obj(),
            client=mock_client,
        )

        assert isinstance(result, EmbeddingResponse)

    @patch(
        "litellm.llms.fabric.embed.handler.get_fabric_auth_header",
        return_value=MOCK_AUTH_HEADER,
    )
    @patch(
        "litellm.llms.fabric.embed.handler.get_fabric_endpoint",
        return_value=MOCK_ENDPOINT,
    )
    def test_embedding_sends_mwctoken_auth(self, mock_endpoint, mock_auth):
        handler = FabricEmbeddingHandler()
        mock_client = _mock_http_client(MOCK_EMBEDDING_RESPONSE)

        handler.embedding(
            model="text-embedding-ada-002",
            input=["Hello world"],
            model_response=EmbeddingResponse(),
            optional_params={},
            logging_obj=_mock_logging_obj(),
            client=mock_client,
        )

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers")
        assert headers["Authorization"] == MOCK_AUTH_HEADER

    def test_embedding_raises_on_missing_fabric_sdk(self):
        handler = FabricEmbeddingHandler()
        with pytest.raises(FabricError) as exc_info:
            handler.embedding(
                model="text-embedding-ada-002",
                input=["Hello world"],
                model_response=EmbeddingResponse(),
                optional_params={},
                logging_obj=_mock_logging_obj(),
            )
        assert exc_info.value.status_code == 401

    @patch(
        "litellm.llms.fabric.embed.handler.get_fabric_auth_header",
        return_value=MOCK_AUTH_HEADER,
    )
    @patch(
        "litellm.llms.fabric.embed.handler.get_fabric_endpoint",
        return_value=MOCK_ENDPOINT,
    )
    def test_embedding_sends_input_in_payload(self, mock_endpoint, mock_auth):
        handler = FabricEmbeddingHandler()
        mock_client = _mock_http_client(MOCK_EMBEDDING_RESPONSE)
        test_input = ["Hello world", "Second sentence"]

        handler.embedding(
            model="text-embedding-ada-002",
            input=test_input,
            model_response=EmbeddingResponse(),
            optional_params={},
            logging_obj=_mock_logging_obj(),
            client=mock_client,
        )

        call_args = mock_client.post.call_args
        sent_data = json.loads(call_args.kwargs.get("data", "{}"))
        assert sent_data["input"] == test_input
