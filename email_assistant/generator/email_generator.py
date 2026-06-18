"""Generates emails using Groq API for Model A and Model B."""

import logging
import os
import re
from groq import Groq
from dotenv import load_dotenv
from generator.prompt_builder import build_model_a_messages, build_model_b_messages

load_dotenv()
logger = logging.getLogger(__name__)

MODEL_A_ID = "llama-3.3-70b-versatile"
MODEL_B_ID = "qwen/qwen3-32b"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        _client = Groq(api_key=api_key)
    return _client


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _clean_output(text: str) -> str:
    """Remove model meta-commentary and markdown formatting from email output."""
    # Drop trailing separator + Note/Disclaimer blocks some models append
    text = re.sub(r"\n[-*]{3,}.*", "", text, flags=re.DOTALL)
    # Strip markdown bold markers (**text** → text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    return text.strip()


def _call_groq(messages: list[dict], model: str, temperature: float = 0.7, max_tokens: int = 900) -> str:
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content.strip()
        content = _strip_thinking(content)
        return _clean_output(content)
    except Exception as e:
        logger.error(f"Groq API error for model {model}: {e}")
        raise


def generate_model_a(intent: str, facts: list[str], tone: str) -> str:
    """Generate email using Model A (advanced prompting: role + few-shot)."""
    messages = build_model_a_messages(intent, facts, tone)
    return _call_groq(messages, MODEL_A_ID)


def generate_model_b(intent: str, facts: list[str], tone: str) -> str:
    """Generate email using Model B (baseline prompting: simple instruction)."""
    messages = build_model_b_messages(intent, facts, tone)
    return _call_groq(messages, MODEL_B_ID)


def generate_email(intent: str, facts: list[str], tone: str, model: str = "A") -> str:
    """Generate email with the specified model ('A' or 'B')."""
    if model.upper() == "A":
        return generate_model_a(intent, facts, tone)
    elif model.upper() == "B":
        return generate_model_b(intent, facts, tone)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'A' or 'B'.")
