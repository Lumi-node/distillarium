"""Teacher backends: pluggable LLM providers that generate the Mash."""

from distillarium.teacher.base import Teacher, DistillExample
from distillarium.teacher.gemini import GeminiTeacher

__all__ = ["Teacher", "DistillExample", "GeminiTeacher"]


def get_teacher(provider: str, **kwargs) -> Teacher:
    """Factory: provider name → Teacher instance."""
    provider = provider.lower()
    if provider == "gemini":
        return GeminiTeacher(**kwargs)
    raise ValueError(
        f"Unknown teacher provider: {provider!r}. Supported: gemini"
    )
