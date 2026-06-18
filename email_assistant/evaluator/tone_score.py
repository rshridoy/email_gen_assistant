"""
Metric 2: Tone Alignment Score (TAS)
LLM-as-a-Judge: rates tone alignment 0-100.
"""

import os
import re
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_PROMPT = """You are an expert communication analyst specialising in tone evaluation.

Expected Tone: {tone}

Generated Email:
{email}

Evaluate how accurately the email matches the expected tone.

Rate the tone alignment from 0 to 100, where:
- 0-20: Completely wrong tone
- 21-40: Mostly wrong, minor elements match
- 41-60: Partially correct, noticeable mismatches
- 61-80: Mostly correct, minor deviations
- 81-100: Excellent tone match throughout

Return ONLY a single integer score with no explanation."""


def compute_tas(generated_email: str, requested_tone: str) -> float:
    """
    Compute Tone Alignment Score using LLM-as-a-Judge.
    Returns a score from 0.0 to 100.0.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = JUDGE_PROMPT.format(tone=requested_tone, email=generated_email)

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"\d+", raw)
        score = int(match.group()) if match else 50
        score = max(0, min(100, score))
        return float(score)
    except Exception as e:
        logger.error(f"TAS judge error: {e}")
        return 50.0
