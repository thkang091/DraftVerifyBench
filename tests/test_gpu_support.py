from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import draftverifybench.utils as utils
import scripts.run_gpu_validation as gpu_validation
from draftverifybench.config import load_config
from draftverifybench.runner import _write_metadata, run_benchmark
from draftverifybench.utils import synchronize_device
from tests.helpers import TinyBundle


def test_gpu_configs_load() -> None:
    smoke = load_config("configs/gpu_smoke.yaml")
    llama = load_config("configs/gpu_llama_1b_8b.yaml")
    reduced = load_config("configs/gpu_llama_reduced.yaml")
    assert smoke.device == "cuda"
    assert llama.device == "cuda"
    assert reduced.validation_level == "gpu_reduced"
    assert llama.batch_size == 1
    assert 8 in llama.draft_ks


def test_cuda_metadata_with_mocked_cuda(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "Mock GPU",
            device_count=lambda: 1,
        ),
        version=SimpleNamespace(cuda="12.4"),
    )
    monkeypatch.setattr(utils, "torch", fake_torch)
    metadata = utils.cuda_hardware_metadata()
    assert metadata["gpu_name"] == "Mock GPU"
    assert metadata["cuda_version"] == "12.4"


def test_synchronize_device_safe_on_cpu() -> None:
    synchronize_device("cpu")


def test_cuda_memory_snapshot_ignores_cpu_device_when_cuda_is_available(monkeypatch) -> None:
    def fail_on_memory_stats(device):
        raise AssertionError(f"CUDA memory stats should not be queried for {device}")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            memory_allocated=fail_on_memory_stats,
            memory_reserved=fail_on_memory_stats,
            max_memory_allocated=fail_on_memory_stats,
        )
    )
    monkeypatch.setattr(utils, "torch", fake_torch)

    snapshot = utils.cuda_memory_snapshot(SimpleNamespace(type="cpu"))

    assert snapshot == {
        "gpu_memory_allocated_bytes": None,
        "gpu_memory_reserved_bytes": None,
        "gpu_max_memory_allocated_bytes": None,
    }


def test_metadata_writer(tmp_path: Path) -> None:
    out = tmp_path / "metadata.json"
    _write_metadata(out, {"device": "cpu", "batch_size": 1})
    assert "batch_size" in out.read_text(encoding="utf-8")


def test_gpu_runbook_exists() -> None:
    assert Path("docs/GPU_Runbook.md").exists()
    assert Path("docs/GPU_Validation_Runbook.md").exists()


def test_run_gpu_validation_skips_without_cuda(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gpu_validation.torch.cuda, "is_available", lambda: False)

    code = gpu_validation.run_gpu_validation(
        "configs/gpu_smoke.yaml",
        out=str(tmp_path / "results.csv"),
        raw_out=str(tmp_path / "raw.jsonl"),
        metadata_out=str(tmp_path / "metadata.json"),
        summary_out=str(tmp_path / "summary.md"),
        require_cuda=False,
    )
    assert code == 0
    assert "skipped_cuda_unavailable" in (tmp_path / "metadata.json").read_text()


def test_run_gpu_validation_require_cuda_fails_without_cuda(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gpu_validation.torch.cuda, "is_available", lambda: False)

    code = gpu_validation.run_gpu_validation(
        "configs/gpu_smoke.yaml",
        out=str(tmp_path / "results.csv"),
        raw_out=str(tmp_path / "raw.jsonl"),
        metadata_out=str(tmp_path / "metadata.json"),
        summary_out=str(tmp_path / "summary.md"),
        require_cuda=True,
    )
    assert code == 2
    assert "failed_require_cuda" in (tmp_path / "summary.md").read_text()


def test_result_rows_include_validation_level(monkeypatch, tmp_path: Path) -> None:
    import draftverifybench.runner as runner

    def fake_loader(model_name: str, **kwargs):
        del kwargs
        return TinyBundle(model_name, [2, 3, 99])

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
validation_level: gpu_smoke
""",
        encoding="utf-8",
    )
    rows = run_benchmark(
        config_path,
        out=tmp_path / "results.csv",
        raw_out=tmp_path / "raw.jsonl",
        metadata_out=tmp_path / "metadata.json",
        max_prompts=1,
    )
    assert rows[0]["validation_level"] == "gpu_smoke"
    assert "latency_ms" in rows[0]
    assert "gpu_memory_allocated_mb" in rows[0]


def test_readme_contains_local_vs_gpu_caveat() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Local Mac/CPU/MPS runs are useful for correctness and debugging" in readme
