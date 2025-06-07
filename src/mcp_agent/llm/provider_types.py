"""
Type definitions for LLM providers.
"""

from enum import Enum


class Provider(Enum):
    """Supported LLM providers"""

    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    FAST_AGENT = "fast-agent"
    GENERIC = "generic"
    GITHUB = "github"  # GitHub Models
    GOOGLE_OAI = "googleoai"  # For Google through OpenAI libraries
    GOOGLE = "google"  # For Google GenAI native library
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    TENSORZERO = "tensorzero"  # For TensorZero Gateway
    AZURE = "azure"  # Azure OpenAI Service
