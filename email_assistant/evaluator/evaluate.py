"""
Evaluation pipeline: runs all 10 scenarios through Model A and Model B,
calculates FIS, TAS, PESS, Final Score, and saves reports.

Usage:
  cd email_assistant
  python evaluator/evaluate.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.email_generator import generate_email
from evaluator.fact_score import compute_fis
from evaluator.tone_score import compute_tas
from evaluator.structure_score import compute_pess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

METRIC_DEFINITIONS = {
    "FIS": (
        "Fact Inclusion Score: measures what percentage of the required key facts "
        "appear in the generated email. Uses keyword matching where a fact is counted "
        "as included if >=60% of its meaningful keywords are present. "
        "Formula: (Matched Facts / Total Facts) x 100. Range: 0-100."
    ),
    "TAS": (
        "Tone Alignment Score: uses an LLM-as-a-Judge approach to rate how accurately "
        "the generated email matches the requested tone. The judge model scores alignment "
        "0-100 based on stylistic, lexical, and emotional tone cues. Range: 0-100."
    ),
    "PESS": (
        "Professional Email Structure Score: evaluates the presence of 5 structural "
        "components (Greeting, Introduction, Main Body, Call-to-Action, Closing), each "
        "worth 20 points. Automated via regex and heuristic checks. Range: 0-100."
    ),
}


def compute_final_score(fis: float, tas: float, pess: float) -> float:
    """Weighted composite: 40% FIS + 30% TAS + 30% PESS."""
    return round(0.4 * fis + 0.3 * tas + 0.3 * pess, 2)


def load_test_cases() -> list[dict]:
    path = DATASETS_DIR / "test_cases.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_one(scenario: dict, model: str) -> dict:
    """Generate and score one scenario with one model."""
    intent = scenario["intent"]
    facts = scenario["facts"]
    tone = scenario["tone"]

    logger.info(f"  Generating: Scenario {scenario['id']} | Model {model}")
    generated = generate_email(intent, facts, tone, model=model)
    time.sleep(1.5)  # Groq free-tier rate limit buffer

    fis = compute_fis(generated, facts)
    time.sleep(0.5)

    tas = compute_tas(generated, tone)
    time.sleep(1.0)

    pess_detail = compute_pess(generated)
    pess = pess_detail["total"]

    final = compute_final_score(fis, tas, pess)

    model_name = "llama-3.3-70b-versatile" if model == "A" else "qwen/qwen3-32b"

    return {
        "ScenarioID": scenario["id"],
        "Category": scenario["category"],
        "Model": f"Model {model} ({model_name})",
        "ModelKey": model,
        "Intent": intent[:60],
        "Tone": tone,
        "GeneratedEmail": generated,
        "FIS": fis,
        "TAS": tas,
        "PESS": pess,
        "PESS_Greeting": pess_detail["greeting"],
        "PESS_Introduction": pess_detail["introduction"],
        "PESS_MainBody": pess_detail["main_body"],
        "PESS_CTA": pess_detail["call_to_action"],
        "PESS_Closing": pess_detail["closing"],
        "FinalScore": final,
    }


def run_evaluation() -> pd.DataFrame:
    """Run full evaluation: 10 scenarios x 2 models."""
    test_cases = load_test_cases()
    all_results = []

    for model_key in ["A", "B"]:
        model_label = (
            "A (llama-3.3-70b-versatile)" if model_key == "A"
            else "B (qwen/qwen3-32b)"
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Model {model_label}")
        logger.info(f"{'='*60}")

        for scenario in tqdm(test_cases, desc=f"Model {model_key}"):
            result = evaluate_one(scenario, model_key)
            all_results.append(result)
            logger.info(
                f"    FIS={result['FIS']:.1f} | TAS={result['TAS']:.1f} | "
                f"PESS={result['PESS']:.1f} | Final={result['FinalScore']:.1f}"
            )

    return pd.DataFrame(all_results)


def save_results(df: pd.DataFrame) -> pd.DataFrame:
    """Save all three report files and return the summary DataFrame."""
    # evaluation_results.csv
    results_cols = ["ScenarioID", "Category", "Model", "FIS", "TAS", "PESS", "FinalScore"]
    df[results_cols].to_csv(REPORTS_DIR / "evaluation_results.csv", index=False)
    logger.info("Saved: reports/evaluation_results.csv")

    # evaluation_summary.csv
    summary = (
        df.groupby("Model")
        .agg(
            AvgFIS=("FIS", "mean"),
            AvgTAS=("TAS", "mean"),
            AvgPESS=("PESS", "mean"),
            AvgFinalScore=("FinalScore", "mean"),
        )
        .round(2)
        .reset_index()
    )
    summary.to_csv(REPORTS_DIR / "evaluation_summary.csv", index=False)
    logger.info("Saved: reports/evaluation_summary.csv")

    # evaluation_report.json
    def model_stats(model_key: str) -> dict:
        sub = df[df["ModelKey"] == model_key]
        return {
            "average_fis": round(float(sub["FIS"].mean()), 2),
            "average_tas": round(float(sub["TAS"].mean()), 2),
            "average_pess": round(float(sub["PESS"].mean()), 2),
            "average_final_score": round(float(sub["FinalScore"].mean()), 2),
            "scenarios": [
                {
                    "scenario_id": int(row["ScenarioID"]),
                    "category": row["Category"],
                    "fis": float(row["FIS"]),
                    "tas": float(row["TAS"]),
                    "pess": float(row["PESS"]),
                    "final_score": float(row["FinalScore"]),
                }
                for _, row in sub.iterrows()
            ],
        }

    report = {
        "metric_definitions": METRIC_DEFINITIONS,
        "model_a": model_stats("A"),
        "model_b": model_stats("B"),
    }

    with open(REPORTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved: reports/evaluation_report.json")

    return summary


def print_summary(summary: pd.DataFrame, df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE — FINAL SUMMARY")
    print("=" * 65)
    print(summary.to_string(index=False))
    print("=" * 65)
    best_model = summary.loc[summary["AvgFinalScore"].idxmax(), "Model"]
    print(f"\nBest performing model: {best_model}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")

    df = run_evaluation()
    summary = save_results(df)
    print_summary(summary, df)
