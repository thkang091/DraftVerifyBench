from __future__ import annotations

from pydantic import BaseModel


class PromptExample(BaseModel):
    prompt_id: str
    prompt_type: str
    prompt_text: str
    expected_entropy_level: str


PROMPTS: list[PromptExample] = [
    PromptExample(
        prompt_id="code_001",
        prompt_type="code_completion",
        prompt_text="Write a Python function that checks if a string is a palindrome.",
        expected_entropy_level="medium",
    ),
    PromptExample(
        prompt_id="code_002",
        prompt_type="code_completion",
        prompt_text="Complete this function: def fibonacci(n):",
        expected_entropy_level="low",
    ),
    PromptExample(
        prompt_id="code_003",
        prompt_type="code_completion",
        prompt_text="Write a SQL query to select users created after 2024.",
        expected_entropy_level="low",
    ),
    PromptExample(
        prompt_id="json_001",
        prompt_type="structured_json",
        prompt_text=(
            "Return a JSON object with name, date, total, and items for this synthetic "
            "receipt text: Alex bought pens and notebooks on 2025-02-01 for 18.42."
        ),
        expected_entropy_level="low",
    ),
    PromptExample(
        prompt_id="json_002",
        prompt_type="structured_json",
        prompt_text=(
            "Convert this short profile into JSON: Maya Chen is a designer in Austin "
            "who likes cycling."
        ),
        expected_entropy_level="low",
    ),
    PromptExample(
        prompt_id="json_003",
        prompt_type="structured_json",
        prompt_text=(
            "Extract invoice fields into JSON: Invoice INV-1007 from Northwind Labs, "
            "due 2025-03-12, total 490.00."
        ),
        expected_entropy_level="low",
    ),
    PromptExample(
        prompt_id="qa_001",
        prompt_type="factual_qa",
        prompt_text="Explain what a hash table is.",
        expected_entropy_level="medium",
    ),
    PromptExample(
        prompt_id="qa_002",
        prompt_type="factual_qa",
        prompt_text="What is gradient descent?",
        expected_entropy_level="medium",
    ),
    PromptExample(
        prompt_id="qa_003",
        prompt_type="factual_qa",
        prompt_text="Summarize the difference between TCP and UDP.",
        expected_entropy_level="medium",
    ),
    PromptExample(
        prompt_id="sum_001",
        prompt_type="summarization",
        prompt_text=(
            "Summarize this synthetic meeting note in two sentences: The team reviewed "
            "latency regressions, assigned profiling work, and postponed the launch "
            "decision until Friday."
        ),
        expected_entropy_level="medium",
    ),
    PromptExample(
        prompt_id="sum_002",
        prompt_type="summarization",
        prompt_text=(
            "Summarize this synthetic policy memo: Employees may expense local transit "
            "for client meetings, but international travel requires manager approval."
        ),
        expected_entropy_level="medium",
    ),
    PromptExample(
        prompt_id="sum_003",
        prompt_type="summarization",
        prompt_text=(
            "Summarize this short technical paragraph: Speculative decoding uses a "
            "smaller draft model to propose tokens and a larger verifier to accept or "
            "reject them."
        ),
        expected_entropy_level="medium",
    ),
    PromptExample(
        prompt_id="open_001",
        prompt_type="open_ended",
        prompt_text="Write a creative story about a robot gardener.",
        expected_entropy_level="high",
    ),
    PromptExample(
        prompt_id="open_002",
        prompt_type="open_ended",
        prompt_text="Brainstorm startup ideas for students.",
        expected_entropy_level="high",
    ),
    PromptExample(
        prompt_id="open_003",
        prompt_type="open_ended",
        prompt_text="Write a persuasive paragraph about remote work.",
        expected_entropy_level="high",
    ),
]


def get_prompts(
    prompt_types: list[str] | None = None,
    max_prompts: int | None = None,
) -> list[PromptExample]:
    prompts = PROMPTS
    if prompt_types:
        allowed = set(prompt_types)
        prompts = [prompt for prompt in prompts if prompt.prompt_type in allowed]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts
