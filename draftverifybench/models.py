from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from draftverifybench.utils import set_seed


def detect_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(device: torch.device, dtype: str = "auto") -> torch.dtype:
    if dtype != "auto":
        return getattr(torch, dtype)
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


@dataclass(frozen=True)
class ModelBundle:
    name: str
    tokenizer: object
    model: torch.nn.Module
    device: torch.device
    dtype: torch.dtype
    parameter_count: int


def load_model_bundle(
    model_name: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    seed: int | None = None,
    local_files_only: bool = False,
    torch_compile: bool = False,
) -> ModelBundle:
    """Load a local Hugging Face causal LM and tokenizer.

    Defaults are intentionally small-model friendly. This function does not choose large models
    implicitly; callers must name every model they want to load.
    """
    set_seed(seed)
    resolved_device = detect_device(device)
    resolved_dtype = select_dtype(resolved_device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolved_dtype,
        local_files_only=local_files_only,
    )
    model.to(resolved_device)
    model.eval()
    if torch_compile and resolved_device.type == "cuda":
        model = torch.compile(model)

    return ModelBundle(
        name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        dtype=resolved_dtype,
        parameter_count=count_parameters(model),
    )
