"""
Microsoft Fabric Azure OpenAI chat completion handler.

Uses raw httpx for HTTP calls because Fabric requires a custom auth scheme
(MwcToken) that the OpenAI Python SDK does not support.
"""

import json
from typing import Any, Callable, Optional, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.databricks.streaming_utils import ModelResponseIterator
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

from ..common_utils import (
    FabricError,
    build_fabric_url,
    get_fabric_api_version,
    get_fabric_auth_header,
    get_fabric_endpoint,
)
from .transformation import FabricConfig


async def _make_async_call(
    client: Optional[AsyncHTTPHandler],
    url: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj: Any,
) -> Any:
    if client is None:
        client = litellm.module_level_aclient

    response = await client.post(url, headers=headers, data=data, stream=True)

    completion_stream = ModelResponseIterator(
        streaming_response=response.aiter_lines(), sync_stream=False
    )

    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


def _make_sync_call(
    client: Optional[HTTPHandler],
    url: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj: Any,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> Any:
    if client is None:
        client = litellm.module_level_client

    response = client.post(
        url, headers=headers, data=data, stream=True, timeout=timeout
    )

    if response.status_code != 200:
        raise FabricError(status_code=response.status_code, message=response.read())

    completion_stream = ModelResponseIterator(
        streaming_response=response.iter_lines(), sync_stream=True
    )

    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class FabricChatCompletion:
    def __init__(self) -> None:
        self.config = FabricConfig()

    def _get_request_params(
        self,
        model: str,
        messages: list,
        api_base: Optional[str],
        api_version: Optional[str],
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple:
        """Build URL, headers, and payload for a Fabric API call.

        Returns:
            Tuple of (url, headers, data_dict)
        """
        auth_header = get_fabric_auth_header()
        endpoint = get_fabric_endpoint(api_base)
        version = get_fabric_api_version(model, api_version)
        url = build_fabric_url(endpoint, model, version, "chat/completions")

        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

        # Use FabricConfig to transform optional_params
        non_default_params = {k: v for k, v in optional_params.items()}
        mapped_params = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=litellm_params.get("drop_params", False) or False,
            api_version=version,
        )

        data = {
            "messages": messages,
            **mapped_params,
        }

        return url, headers, data

    async def _acompletion(
        self,
        model: str,
        messages: list,
        url: str,
        headers: dict,
        data: dict,
        model_response: ModelResponse,
        logging_obj: Any,
        client: Optional[AsyncHTTPHandler] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> ModelResponse:
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
        return ModelResponse(**response_json)

    async def _acompletion_stream(
        self,
        model: str,
        messages: list,
        url: str,
        headers: dict,
        data: dict,
        logging_obj: Any,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> CustomStreamWrapper:
        data["stream"] = True
        completion_stream = await _make_async_call(
            client=client,
            url=url,
            headers=headers,
            data=json.dumps(data),
            model=model,
            messages=messages,
            logging_obj=logging_obj,
        )
        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="fabric",
            logging_obj=logging_obj,
        )

    def completion(
        self,
        *,
        model: str,
        messages: list,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        model_response: ModelResponse,
        print_verbose: Optional[Callable] = None,
        optional_params: dict,
        litellm_params: dict = {},
        logger_fn: Optional[Callable] = None,
        encoding: Any = None,
        logging_obj: Any,
        acompletion: Optional[bool] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        headers: Optional[dict] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        stream: bool = optional_params.pop("stream", False) or False
        optional_params.pop("max_retries", None)

        url, fabric_headers, data = self._get_request_params(
            model=model,
            messages=messages,
            api_base=api_base,
            api_version=api_version,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        # Merge any extra headers
        if headers:
            fabric_headers.update(headers)

        # Logging
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": fabric_headers,
            },
        )

        if acompletion is True:
            async_client = client if isinstance(client, AsyncHTTPHandler) else None
            if stream:
                data["stream"] = True
                return self._acompletion_stream(
                    model=model,
                    messages=messages,
                    url=url,
                    headers=fabric_headers,
                    data=data,
                    logging_obj=logging_obj,
                    client=async_client,
                )
            else:
                return self._acompletion(
                    model=model,
                    messages=messages,
                    url=url,
                    headers=fabric_headers,
                    data=data,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    client=async_client,
                    timeout=timeout,
                )
        else:
            sync_client = client if isinstance(client, HTTPHandler) else None
            if stream:
                data["stream"] = True
                completion_stream = _make_sync_call(
                    client=sync_client,
                    url=url,
                    headers=fabric_headers,
                    data=json.dumps(data),
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                    timeout=timeout,
                )
                return CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider="fabric",
                    logging_obj=logging_obj,
                )
            else:
                if sync_client is None:
                    sync_client = HTTPHandler(timeout=timeout)
                try:
                    response = sync_client.post(
                        url=url,
                        headers=fabric_headers,
                        data=json.dumps(data),
                    )
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise FabricError(
                        status_code=e.response.status_code,
                        message=e.response.text,
                    )
                except httpx.TimeoutException:
                    raise FabricError(
                        status_code=408, message="Timeout error occurred."
                    )
                except Exception as e:
                    if isinstance(e, FabricError):
                        raise
                    raise FabricError(status_code=500, message=str(e))

                response_json = response.json()
                return ModelResponse(**response_json)
