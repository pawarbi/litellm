"""
Fabric Azure OpenAI chat completion configuration.

Reuses Azure OpenAI parameter transformation logic with Fabric-specific
adjustments for reasoning models.
"""

from typing import List

from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig

from ..common_utils import FabricError, is_fabric_reasoning_model


class FabricConfig(AzureOpenAIConfig):
    """
    Configuration for Microsoft Fabric Azure OpenAI chat completions.

    Fabric provides built-in Azure OpenAI endpoints in notebook environments.
    This config reuses Azure OpenAI parameter transformation with adjustments
    for reasoning models (gpt-5 family).

    Reference: https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview
    """

    def get_supported_openai_params(self, model: str) -> List[str]:
        base_params = super().get_supported_openai_params(model)

        if is_fabric_reasoning_model(model):
            # Reasoning models do not support temperature or n
            unsupported = {"temperature", "n"}
            base_params = [p for p in base_params if p not in unsupported]
            if "max_completion_tokens" not in base_params:
                base_params.append("max_completion_tokens")

        return base_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        api_version: str = "",
    ) -> dict:
        if is_fabric_reasoning_model(model):
            # For reasoning models, convert max_tokens to max_completion_tokens
            if "max_tokens" in non_default_params:
                non_default_params = non_default_params.copy()
                non_default_params["max_completion_tokens"] = non_default_params.pop(
                    "max_tokens"
                )

            # Remove temperature and n if present
            non_default_params = {
                k: v
                for k, v in non_default_params.items()
                if k not in ("temperature", "n")
            }

        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
            api_version=api_version,
        )

    def get_error_class(self) -> type:
        return FabricError
