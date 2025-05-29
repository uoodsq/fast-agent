from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

GITHUB_BASE_URL = "https://models.github.ai/inference"
DEFAULT_GITHUB_MODEL = "openai/gpt-4.1"


class GitHubAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["provider"] = Provider.GITHUB

        super().__init__(*args, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize GitHub-specific default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_GITHUB_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _base_url(self) -> str:
        base_url = None

        if self.context.config and self.context.config.github:
            base_url = self.context.config.github.base_url

        return base_url if base_url else GITHUB_BASE_URL
