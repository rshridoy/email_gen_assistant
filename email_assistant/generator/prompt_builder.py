"""Builds prompts for Model A (advanced) and Model B (baseline)."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_system_prompt() -> str:
    path = PROMPTS_DIR / "system_prompt.txt"
    return path.read_text(encoding="utf-8").strip()


def _load_few_shot_examples() -> list[dict[str, Any]]:
    path = PROMPTS_DIR / "few_shot_examples.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _format_facts(facts: list[str]) -> str:
    return "\n".join(f"  - {fact}" for fact in facts)


def _format_few_shot_block(examples: list[dict[str, Any]]) -> str:
    blocks = []
    for i, ex in enumerate(examples, 1):
        inp = ex["input"]
        facts_str = _format_facts(inp["facts"])
        block = (
            f"--- EXAMPLE {i} ---\n"
            f"Intent: {inp['intent']}\n"
            f"Key Facts:\n{facts_str}\n"
            f"Tone: {inp['tone']}\n\n"
            f"EMAIL:\n{ex['output']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def build_model_a_messages(intent: str, facts: list[str], tone: str) -> list[dict[str, str]]:
    """Advanced prompt: role prompting + few-shot examples."""
    system_prompt = _load_system_prompt()
    examples = _load_few_shot_examples()
    few_shot_block = _format_few_shot_block(examples)
    facts_str = _format_facts(facts)

    system_content = (
        f"{system_prompt}\n\n"
        f"Study these examples carefully to understand the expected quality and format:\n\n"
        f"{few_shot_block}\n\n"
        f"Now generate an email for the following input. Follow the same format exactly."
    )

    user_content = (
        f"Intent: {intent}\n"
        f"Key Facts:\n{facts_str}\n"
        f"Tone: {tone}\n\n"
        f"EMAIL:"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_model_b_messages(intent: str, facts: list[str], tone: str) -> list[dict[str, str]]:
    """Baseline prompt: simple instruction only, no role, no examples."""
    facts_str = _format_facts(facts)
    user_content = (
        f"Write a professional email using the following information.\n\n"
        f"Intent: {intent}\n"
        f"Key Facts:\n{facts_str}\n"
        f"Tone: {tone}"
    )
    return [{"role": "user", "content": user_content}]
