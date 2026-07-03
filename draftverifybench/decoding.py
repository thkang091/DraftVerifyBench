from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel

from draftverifybench.utils import elapsed_ms, now_perf, set_seed, synchronize_device


class BaselineDecodeResult(BaseModel):
    output_token_ids: list[int]
    output_text: str
    generated_tokens: int
    total_latency_ms: float
    time_to_first_token_ms: float | None
    tokens_per_second: float
    per_token_latency_ms: list[float]
    verifier_forward_calls: int


def _device_of(bundle: Any) -> torch.device:
    return getattr(bundle, "device", torch.device("cpu"))


def _decode(tokenizer: Any, ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(str(token_id) for token_id in ids)


def _encode(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return input_ids.to(device)


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    logits = logits.float()
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        keep = min(top_k, logits.shape[-1])
        values, _ = torch.topk(logits, keep)
        logits = torch.where(
            logits < values[..., -1, None],
            torch.full_like(logits, -float("inf")),
            logits,
        )

    probs = F.softmax(logits, dim=-1)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_indices.gather(-1, sampled_index).item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def next_token_id(
    bundle: Any,
    input_ids: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    if hasattr(bundle.model, "next_token"):
        return int(bundle.model.next_token(input_ids))
    outputs = bundle.model(input_ids=input_ids)
    logits = outputs.logits[:, -1, :]
    return sample_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)


def baseline_decode(
    bundle: Any,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> BaselineDecodeResult:
    set_seed(seed)
    device = _device_of(bundle)
    input_ids = _encode(bundle.tokenizer, prompt, device)
    prompt_len = input_ids.shape[-1]
    eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
    per_token_latency_ms: list[float] = []
    verifier_forward_calls = 0
    first_token_ms: float | None = None
    start = now_perf()
    synchronize_device(device)

    for _ in range(max_new_tokens):
        synchronize_device(device)
        token_start = now_perf()
        token_id = next_token_id(
            bundle,
            input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        verifier_forward_calls += 1
        synchronize_device(device)
        per_token_latency_ms.append(elapsed_ms(token_start))
        if first_token_ms is None:
            first_token_ms = elapsed_ms(start)
        next_token = torch.tensor([[token_id]], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if eos_token_id is not None and token_id == eos_token_id:
            break

    synchronize_device(device)
    total_ms = elapsed_ms(start)
    generated_ids = input_ids[0, prompt_len:].detach().cpu().tolist()
    generated = len(generated_ids)
    return BaselineDecodeResult(
        output_token_ids=generated_ids,
        output_text=_decode(bundle.tokenizer, generated_ids),
        generated_tokens=generated,
        total_latency_ms=total_ms,
        time_to_first_token_ms=first_token_ms,
        tokens_per_second=(generated / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
        per_token_latency_ms=per_token_latency_ms,
        verifier_forward_calls=verifier_forward_calls,
    )
