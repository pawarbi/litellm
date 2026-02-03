"""
Microsoft Fabric Azure OpenAI embedding handler.

Uses raw httpx for HTTP calls because Fabric requires a custom auth scheme
(MwcToken) that the OpenAI Python SDK does not support.
"""

import json
from typing import Any, Optional, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import EmbeddingResponse

from ..common_utils import (
    FabricError,
    build_fabric_url,
    get_fabric_api_version,
    get_fabric_auth_header,
    get_fabric_endpoint,
)


class FabricEmbeddingHandler:
    def embedding(
        self,
        *,
        model: str,
        input: list,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        model_response: EmbeddingResponse,
        optional_params: dict,
        logging_obj: Any,
        aembedding: Optional[bool] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ) -> EmbeddingResponse:
        auth_header = get_fabric_auth_header()
        endpoint = get_fabric_endpoint(api_base)
        version = get_fabric_api_version(model, api_version)
        url = build_fabric_url(endpoint, model, version, "embeddings")

        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

        data = {"input": input}
        if optional_params:
            data.update(optional_params)

        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )

        if aembedding is True:
            return self._aembedding(
                url=url,
                headers=headers,
                data=data,
                model_response=model_response,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        sync_client = (
            client if isinstance(client, HTTPHandler) else HTTPHandler(timeout=timeout)
        )

        try:
            response = sync_client.post(
                url=url,
                headers=headers,
                data=json.dumps(data),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise FabricError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except httpx.TimeoutException:
            raise FabricError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            if isinstance(e, FabricError):
                raise
            raise FabricError(status_code=500, message=str(e))

        response_json = response.json()
        return EmbeddingResponse(**response_json)

    async def _aembedding(
        self,
        url: str,
        headers: dict,
        data: dict,
        model_response: EmbeddingResponse,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> EmbeddingResponse:
        if timeout is None:
            timeout = httpx.Timeout(timeout=600.0, connect=5.0)

        if client is None:
            client = litellm.module_level_aclient

        try:
            response = await client.post(
                url, headers=headers, data=json.dumps(data), timeout=timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise FabricError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except httpx.TimeoutException:
            raise FabricError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            if isinstance(e, FabricError):
                raise
            raise FabricError(status_code=500, message=str(e))

        response_json = response.json()
        return EmbeddingResponse(**response_json)
