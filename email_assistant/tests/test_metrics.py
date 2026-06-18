"""Unit tests for the three evaluation metrics (FIS and PESS — no API key needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.fact_score import compute_fis
from evaluator.structure_score import compute_pess


class TestFactInclusionScore:
    def test_all_facts_present(self) -> None:
        email = "We met on June 15th to discuss the project. The budget is $50,000. Please confirm by Friday."
        facts = ["June 15th", "project", "budget", "Friday"]
        score = compute_fis(email, facts)
        assert score >= 75.0, f"Expected >=75.0, got {score}"

    def test_no_facts_present(self) -> None:
        email = "Hello, hope you are well."
        facts = ["quarterly report", "stakeholder meeting", "revenue growth"]
        score = compute_fis(email, facts)
        assert score == 0.0, f"Expected 0.0, got {score}"

    def test_empty_facts(self) -> None:
        score = compute_fis("Any email content here.", [])
        assert score == 100.0, f"Expected 100.0, got {score}"

    def test_partial_facts(self) -> None:
        email = "The meeting was on Monday. Please respond soon."
        facts = ["meeting", "Monday", "budget approval", "CFO review"]
        score = compute_fis(email, facts)
        assert 25.0 <= score <= 75.0, f"Expected 25-75, got {score}"

    def test_score_range(self) -> None:
        email = "Subject: Update\n\nDear Team, please find attached the quarterly report.\n\nBest regards,\n[Name]"
        facts = ["quarterly report", "attached"]
        score = compute_fis(email, facts)
        assert 0.0 <= score <= 100.0


class TestProfessionalEmailStructureScore:
    PERFECT_EMAIL = """Subject: Project Update

Dear John,

I am writing to provide you with an update on the project status this week.

The team has made significant progress and we are on track for the deadline. All milestones are being met.

Please let me know if you have any questions or if there is anything else I can help with.

Best regards,
[Your Name]"""

    def test_perfect_email_scores_high(self) -> None:
        result = compute_pess(self.PERFECT_EMAIL)
        assert result["total"] >= 80, f"Expected >=80, got {result['total']}"

    def test_greeting_detected(self) -> None:
        result = compute_pess(self.PERFECT_EMAIL)
        assert result["greeting"] == 20

    def test_cta_detected(self) -> None:
        result = compute_pess(self.PERFECT_EMAIL)
        assert result["call_to_action"] == 20

    def test_closing_detected(self) -> None:
        result = compute_pess(self.PERFECT_EMAIL)
        assert result["closing"] == 20

    def test_missing_greeting(self) -> None:
        email = """Subject: Update

The project is going well and we are on track. Please review and confirm your availability.

Regards,
[Your Name]"""
        result = compute_pess(email)
        assert result["greeting"] == 0

    def test_missing_closing(self) -> None:
        email = """Dear Team,

Please note the project update below.

The deadline is approaching. Let me know your thoughts on the proposal."""
        result = compute_pess(email)
        assert result["closing"] == 0

    def test_scores_are_multiples_of_20(self) -> None:
        email = "Hello, please confirm. Thanks."
        result = compute_pess(email)
        for key in ["greeting", "introduction", "main_body", "call_to_action", "closing"]:
            assert result[key] in [0, 20], f"{key} should be 0 or 20, got {result[key]}"

    def test_total_equals_sum_of_components(self) -> None:
        result = compute_pess(self.PERFECT_EMAIL)
        component_sum = (
            result["greeting"]
            + result["introduction"]
            + result["main_body"]
            + result["call_to_action"]
            + result["closing"]
        )
        assert result["total"] == component_sum

    def test_total_max_100(self) -> None:
        result = compute_pess(self.PERFECT_EMAIL)
        assert result["total"] <= 100.0


if __name__ == "__main__":
    print("Running FIS tests...")
    t1 = TestFactInclusionScore()
    t1.test_all_facts_present()
    t1.test_no_facts_present()
    t1.test_empty_facts()
    t1.test_partial_facts()
    t1.test_score_range()
    print("  All FIS tests passed.")

    print("Running PESS tests...")
    t2 = TestProfessionalEmailStructureScore()
    t2.test_perfect_email_scores_high()
    t2.test_greeting_detected()
    t2.test_cta_detected()
    t2.test_closing_detected()
    t2.test_missing_greeting()
    t2.test_missing_closing()
    t2.test_scores_are_multiples_of_20()
    t2.test_total_equals_sum_of_components()
    t2.test_total_max_100()
    print("  All PESS tests passed.")

    print("\nAll tests passed!")
