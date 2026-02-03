"""
Fabric Azure OpenAI embedding configuration.

Fabric embeddings use standard Azure OpenAI embedding parameters.
No provider-specific parameter transformations are needed.
"""


class FabricEmbeddingConfig:
    """
    Configuration for Microsoft Fabric Azure OpenAI embeddings.

    Reference: https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview
    """

    def get_supported_openai_params(self) -> list:
        return ["encoding_format", "dimensions"]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        for k, v in non_default_params.items():
            if k in self.get_supported_openai_params():
                optional_params[k] = v
        return optional_params
