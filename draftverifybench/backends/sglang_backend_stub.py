from __future__ import annotations


class SGLangBackend:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise NotImplementedError(
            "SGLang backend is a planned integration. Verify SGLang server/runtime APIs for the "
            "target version before enabling this backend."
        )

