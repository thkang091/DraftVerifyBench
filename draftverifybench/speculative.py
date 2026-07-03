from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel

from draftverifybench.decoding import _decode, _device_of, _encode, next_token_id
from draftverifybench.utils import elapsed_ms, now_perf, set_seed, synchronize_device


class SpeculativeDecodeResult(BaseModel):
    output_token_ids: list[int]
    output_text: str
    generated_tokens: int
    total_latency_ms: float
    tokens_per_second: float
    verifier_forward_calls: int
    draft_forward_calls: int
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    draft_tokens_rejected: int
    acceptance_rate: float
    draft_overhead_ms: float


def _append(input_ids: torch.Tensor, token_id: int, device: torch.device) -> torch.Tensor:
    token = torch.tensor([[token_id]], dtype=input_ids.dtype, device=device)
    return torch.cat([input_ids, token], dim=-1)


def _greedy_tokens_for_contexts(
    bundle: Any,
    input_ids: torch.Tensor,
    draft_ids: list[int],
) -> list[int]:
    if not draft_ids:
        return []
    if hasattr(bundle.model, "next_token"):
        context = input_ids
        tokens: list[int] = []
        for draft_id in draft_ids:
            tokens.append(int(bundle.model.next_token(context)))
            context = _append(context, draft_id, _device_of(bundle))
        return tokens

    device = _device_of(bundle)
    prefix = input_ids.to(device)
    if len(draft_ids) > 1:
        draft_prefix = torch.tensor(
            [draft_ids[:-1]],
            dtype=prefix.dtype,
            device=device,
        )
        verifier_input = torch.cat([prefix, draft_prefix], dim=-1)
    else:
        verifier_input = prefix
    outputs = bundle.model(input_ids=verifier_input)
    logits = outputs.logits
    start = prefix.shape[-1] - 1
    end = start + len(draft_ids)
    return torch.argmax(logits[:, start:end, :].float(), dim=-1)[0].detach().cpu().tolist()


@torch.inference_mode()
def speculative_decode(
    draft_bundle: Any,
    verifier_bundle: Any,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    draft_k: int = 4,
    temperature: float = 0.0,
    seed: int | None = None,
) -> SpeculativeDecodeResult:
    """Greedy speculative decoding.

    Temperature sampling is intentionally not implemented here because exact speculative sampling
    requires distribution correction. Non-zero temperatures currently use greedy verification and
    should be treated as experimental diagnostics, not exact sampling.
    """
    set_seed(seed)
    if draft_k < 1:
        raise ValueError("draft_k must be >= 1")

    device = _device_of(verifier_bundle)
    input_ids = _encode(verifier_bundle.tokenizer, prompt, device)
    prompt_len = input_ids.shape[-1]
    eos_token_id = getattr(verifier_bundle.tokenizer, "eos_token_id", None)

    verifier_forward_calls = 0
    draft_forward_calls = 0
    proposed = 0
    accepted = 0
    rejected = 0
    draft_overhead_ms = 0.0
    synchronize_device(device)
    start = now_perf()

    while (input_ids.shape[-1] - prompt_len) < max_new_tokens:
        draft_ids: list[int] = []
        draft_context = input_ids.to(_device_of(draft_bundle))
        for _ in range(draft_k):
            if (input_ids.shape[-1] - prompt_len + len(draft_ids)) >= max_new_tokens:
                break
            synchronize_device(_device_of(draft_bundle))
            draft_start = now_perf()
            draft_token = next_token_id(draft_bundle, draft_context, temperature=0.0)
            synchronize_device(_device_of(draft_bundle))
            draft_overhead_ms += elapsed_ms(draft_start)
            draft_forward_calls += 1
            proposed += 1
            draft_ids.append(draft_token)
            draft_context = _append(draft_context, draft_token, _device_of(draft_bundle))
            if eos_token_id is not None and draft_token == eos_token_id:
                break

        if not draft_ids:
            break

        synchronize_device(device)
        verifier_tokens = _greedy_tokens_for_contexts(verifier_bundle, input_ids, draft_ids)
        synchronize_device(device)
        verifier_forward_calls += 1
        for draft_token, verifier_token in zip(draft_ids, verifier_tokens, strict=True):
            if draft_token == verifier_token:
                accepted += 1
                input_ids = _append(input_ids, draft_token, device)
                if eos_token_id is not None and draft_token == eos_token_id:
                    break
            else:
                rejected += 1
                input_ids = _append(input_ids, verifier_token, device)
                break

            if (input_ids.shape[-1] - prompt_len) >= max_new_tokens:
                break

        if eos_token_id is not None and int(input_ids[0, -1].item()) == eos_token_id:
            break

    synchronize_device(device)
    total_ms = elapsed_ms(start)
    generated_ids = input_ids[0, prompt_len:].detach().cpu().tolist()
    generated = len(generated_ids)
    acceptance_rate = accepted / proposed if proposed else 0.0
    return SpeculativeDecodeResult(
        output_token_ids=generated_ids,
        output_text=_decode(verifier_bundle.tokenizer, generated_ids),
        generated_tokens=generated,
        total_latency_ms=total_ms,
        tokens_per_second=(generated / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
        verifier_forward_calls=verifier_forward_calls,
        draft_forward_calls=draft_forward_calls,
        draft_tokens_proposed=proposed,
        draft_tokens_accepted=accepted,
        draft_tokens_rejected=rejected,
        acceptance_rate=acceptance_rate,
        draft_overhead_ms=draft_overhead_ms,
    )
