from __future__ import annotations


class VLLMBackend:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise NotImplementedError(
            "vLLM backend is a planned integration. Map DraftVerifyBench prompts and metrics to "
            "a verified vLLM version before enabling this backend."
        )

