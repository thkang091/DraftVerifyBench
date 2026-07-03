from __future__ import annotations

from dataclasses import dataclass

import torch


class TinyTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token = "<eos>"

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        del text, return_tensors
        return {"input_ids": torch.tensor([[1]], dtype=torch.long)}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(item) for item in ids)


class SequenceModel:
    def __init__(self, sequence: list[int]) -> None:
        self.sequence = sequence

    def next_token(self, input_ids: torch.Tensor) -> int:
        generated = max(input_ids.shape[-1] - 1, 0)
        if generated >= len(self.sequence):
            return self.sequence[-1]
        return self.sequence[generated]


@dataclass
class TinyBundle:
    name: str
    sequence: list[int]
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    parameter_count: int = 0

    def __post_init__(self) -> None:
        self.tokenizer = TinyTokenizer()
        self.model = SequenceModel(self.sequence)

