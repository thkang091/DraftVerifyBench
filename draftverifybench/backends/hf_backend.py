from __future__ import annotations

from typing import Any

from draftverifybench.models import ModelBundle, load_model_bundle


class HuggingFaceBackend:
    """Working local Hugging Face backend used by DraftVerifyBench."""

    def load(self, model_name: str, **kwargs: Any) -> ModelBundle:
        return load_model_bundle(model_name, **kwargs)

