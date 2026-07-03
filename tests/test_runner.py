from __future__ import annotations

import json
from pathlib import Path

from draftverifybench import runner
from tests.helpers import TinyBundle


def test_runner_writes_csv_jsonl_and_metadata(monkeypatch, tmp_path: Path) -> None:
    def fake_loader(model_name: str, **kwargs):
        del kwargs
        sequence = [2, 3, 99] if model_name == "draft" else [2, 3, 99]
        return TinyBundle(model_name, sequence)

    monkeypatch.setattr(runner, "load_model_bundle", fake_loader)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 1
device: auto
draft_model: draft
verifier_model: verifier
max_new_tokens: 4
temperatures: [0.0]
draft_ks: [2]
prompt_types: [factual_qa]
repetitions: 1
""",
        encoding="utf-8",
    )
    out = tmp_path / "results.csv"
    raw = tmp_path / "raw.jsonl"
    meta = tmp_path / "metadata.json"
    rows = runner.run_benchmark(config_path, out=out, raw_out=raw, metadata_out=meta, max_prompts=1)
    assert out.exists()
    assert raw.exists()
    assert meta.exists()
    assert len(rows) == 2
    assert json.loads(meta.read_text(encoding="utf-8"))["draft_model"] == "draft"

