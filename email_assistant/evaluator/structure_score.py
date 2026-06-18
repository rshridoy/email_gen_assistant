"""
Metric 3: Professional Email Structure Score (PESS)
Evaluates 5 structural components, 20 points each. Max: 100.

Components:
  1. Greeting (20 pts)       - starts with Dear/Hi/Hello/Good morning
  2. Introduction (20 pts)   - first paragraph exists and is >=15 words
  3. Main Body (20 pts)      - at least 2 body paragraphs
  4. Call-to-Action (20 pts) - contains action language
  5. Closing (20 pts)        - ends with Regards/Sincerely/Best/Thank you/etc.
"""

import re
import logging

logger = logging.getLogger(__name__)

GREETING_PATTERNS = [
    r"^\s*(dear|hi|hello|good morning|good afternoon|greetings|to whom)",
]
# Allow closing keyword at end-of-string or followed by newline
CLOSING_PATTERNS = [
    r"(regards|sincerely|best wishes|yours truly|thank you|warm regards|"
    r"kind regards|respectfully|best|cheers)\s*,?\s*(\n|$)",
]
_GREETING_RE = re.compile(r"^(dear|hi|hello|good morning|good afternoon|greetings|to whom)\b", re.IGNORECASE)
CTA_KEYWORDS = [
    "please", "kindly", "let me know", "feel free", "do not hesitate",
    "contact us", "reach out", "reply", "respond", "rsvp", "confirm",
    "schedule", "book", "call", "email", "visit", "click", "submit",
    "looking forward",
]


def _check_greeting(email: str) -> bool:
    first_lines = "\n".join(email.strip().splitlines()[:5])
    return any(re.search(p, first_lines, re.IGNORECASE | re.MULTILINE) for p in GREETING_PATTERNS)


def _check_introduction(email: str) -> bool:
    lines = email.strip().splitlines()
    in_para = False
    para_words: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("subject:"):
            continue
        if _GREETING_RE.match(stripped):
            # Skip greeting line; don't let it count as introduction text
            if in_para and para_words:
                para_words = []
                in_para = False
            continue
        if stripped:
            in_para = True
            para_words.extend(stripped.split())
        elif in_para and para_words:
            break
    return len(para_words) >= 15


def _check_main_body(email: str) -> bool:
    lines = email.strip().splitlines()
    paragraphs = []
    current: list[str] = []
    for line in lines:
        if line.strip().lower().startswith("subject:"):
            continue
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return len(paragraphs) >= 2


def _check_cta(email: str) -> bool:
    lower = email.lower()
    return any(kw in lower for kw in CTA_KEYWORDS)


def _check_closing(email: str) -> bool:
    last_lines = "\n".join(email.strip().splitlines()[-8:])
    return any(re.search(p, last_lines, re.IGNORECASE) for p in CLOSING_PATTERNS)


def compute_pess(generated_email: str) -> dict:
    """
    Compute Professional Email Structure Score.
    Returns dict with individual component scores and total (0-100).
    """
    greeting = 20 if _check_greeting(generated_email) else 0
    introduction = 20 if _check_introduction(generated_email) else 0
    main_body = 20 if _check_main_body(generated_email) else 0
    cta = 20 if _check_cta(generated_email) else 0
    closing = 20 if _check_closing(generated_email) else 0

    total = greeting + introduction + main_body + cta + closing

    logger.debug(
        f"PESS breakdown: greeting={greeting}, intro={introduction}, "
        f"body={main_body}, cta={cta}, closing={closing}, total={total}"
    )

    return {
        "greeting": greeting,
        "introduction": introduction,
        "main_body": main_body,
        "call_to_action": cta,
        "closing": closing,
        "total": float(total),
    }
