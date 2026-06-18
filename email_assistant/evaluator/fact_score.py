"""
Metric 1: Fact Inclusion Score (FIS)
FIS = (Matched Facts / Total Facts) * 100
A fact is "matched" if >=60% of its meaningful keywords appear in the email.
"""

import re
import logging

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def _extract_keywords(fact: str) -> list[str]:
    return [w for w in _normalise(fact).split() if len(w) > 3]


def compute_fis(generated_email: str, facts: list[str]) -> float:
    """
    Compute Fact Inclusion Score.
    Returns a score from 0.0 to 100.0.
    """
    if not facts:
        return 100.0

    email_norm = _normalise(generated_email)
    matched = 0

    for fact in facts:
        keywords = _extract_keywords(fact)
        if not keywords:
            matched += 1
            continue
        hits = sum(1 for kw in keywords if kw in email_norm)
        if hits / len(keywords) >= 0.6:
            matched += 1
            logger.debug(f"MATCHED: {fact}")
        else:
            logger.debug(f"MISSED: {fact} (hits={hits}/{len(keywords)})")

    score = (matched / len(facts)) * 100
    return round(score, 2)
