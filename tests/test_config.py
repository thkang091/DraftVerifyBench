from __future__ import annotations

from draftverifybench.config import load_config


def test_config_loading() -> None:
    config = load_config("configs/local_small.yaml")
    assert config.seed == 42
    assert config.draft_model == "distilgpt2"
    assert 4 in config.draft_ks
    assert "open_ended" in config.prompt_types

