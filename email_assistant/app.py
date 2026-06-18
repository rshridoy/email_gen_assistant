"""
Email Generation Assistant — CLI entry point.

Usage:
  python app.py generate
  python app.py evaluate
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def cmd_generate() -> None:
    from generator.email_generator import generate_email

    print("\n=== Email Generation Assistant ===\n")
    intent = input("Intent: ").strip()
    facts_raw = input("Key Facts (comma-separated): ").strip()
    facts = [f.strip() for f in facts_raw.split(",") if f.strip()]
    tone = input("Tone: ").strip()
    model = input("Model A or B? [A]: ").strip().upper() or "A"

    print(f"\nGenerating email with Model {model}...\n")
    email = generate_email(intent, facts, tone, model=model)
    print("=" * 60)
    print(email)
    print("=" * 60)


def cmd_evaluate() -> None:
    from evaluator.evaluate import run_evaluation, save_results, print_summary

    df = run_evaluation()
    summary = save_results(df)
    print_summary(summary, df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Email Generation Assistant")
    parser.add_argument("command", choices=["generate", "evaluate"], help="Command to run")
    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate()
    elif args.command == "evaluate":
        cmd_evaluate()


if __name__ == "__main__":
    main()
