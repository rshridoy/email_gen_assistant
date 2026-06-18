"""Streamlit frontend for the Email Generation Assistant."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Email Generation Assistant",
    page_icon="✉️",
    layout="wide",
)

st.title("✉️ Email Generation Assistant")
st.caption(
    "Powered by Groq Cloud — "
    "llama-3.3-70b-versatile (Model A · Advanced) vs "
    "qwen-qwq-32b (Model B · Baseline)"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Select Model",
        ["A — llama-3.3-70b (Advanced)", "B — deepseek-r1 (Baseline)"],
    )
    model_key = "A" if model_choice.startswith("A") else "B"

    st.divider()
    st.markdown("**Tone Presets**")
    tone_preset = st.selectbox(
        "Quick select tone",
        [
            "Custom",
            "Professional",
            "Formal",
            "Friendly and professional",
            "Empathetic and sincere",
            "Empathetic and solution-focused",
            "Urgent and formal",
            "Confident and collaborative",
            "Firm and professional",
            "Warm and genuine",
            "Professional and concise",
        ],
    )

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This assistant uses advanced prompt engineering (role prompting + "
        "few-shot examples) for Model A and a simple baseline prompt for Model B."
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_gen, tab_eval = st.tabs(["Generate Email", "Evaluation Dashboard"])

# ── Generate Tab ───────────────────────────────────────────────────────────────
with tab_gen:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Input")
        intent = st.text_input(
            "Intent",
            placeholder="e.g. Follow up after a business meeting to confirm action items",
        )
        facts_raw = st.text_area(
            "Key Facts (one per line)",
            placeholder=(
                "Meeting held on June 15th\n"
                "Discussed AI automation strategy\n"
                "Budget approval needed from CFO"
            ),
            height=160,
        )
        if tone_preset == "Custom":
            tone = st.text_input("Tone", placeholder="e.g. Professional and warm")
        else:
            tone = st.text_input("Tone", value=tone_preset)

        generate_btn = st.button(
            f"Generate Email  (Model {model_key})",
            type="primary",
            use_container_width=True,
        )

    with col2:
        st.subheader("Generated Email")
        placeholder = st.empty()

        if generate_btn:
            if not intent.strip():
                st.error("Please enter an intent.")
            elif not facts_raw.strip():
                st.error("Please enter at least one key fact.")
            elif not tone.strip():
                st.error("Please enter a tone.")
            else:
                facts = [f.strip() for f in facts_raw.strip().splitlines() if f.strip()]
                with st.spinner(f"Generating with Model {model_key}..."):
                    try:
                        from generator.email_generator import generate_email

                        email = generate_email(intent, facts, tone, model=model_key)
                        st.session_state["last_email"] = email
                        st.session_state["last_facts"] = facts
                        st.session_state["last_tone"] = tone
                    except Exception as exc:
                        st.error(f"Generation error: {exc}")

        if "last_email" in st.session_state:
            email_text = st.session_state["last_email"]
            st.text_area(
                "email_output",
                value=email_text,
                height=460,
                label_visibility="collapsed",
            )
            st.download_button(
                "Download Email (.txt)",
                data=email_text,
                file_name="generated_email.txt",
                mime="text/plain",
                use_container_width=True,
            )

# ── Evaluation Dashboard Tab ───────────────────────────────────────────────────
with tab_eval:
    st.subheader("Evaluation Dashboard")
    st.markdown(
        "Run the full evaluation (10 scenarios × 2 models × 3 metrics) or view "
        "previously saved results."
    )

    reports_dir = BASE_DIR / "reports"
    report_path = reports_dir / "evaluation_report.json"
    results_path = reports_dir / "evaluation_results.csv"

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_eval_btn = st.button(
            "Run Full Evaluation",
            type="secondary",
            use_container_width=True,
            help="Runs 20 generation + scoring calls. Takes ~3-4 minutes.",
        )

    if run_eval_btn:
        with st.spinner("Running evaluation on 10 scenarios × 2 models (~3-4 minutes)..."):
            try:
                from evaluator.evaluate import run_evaluation, save_results

                df_eval = run_evaluation()
                save_results(df_eval)
                st.success("Evaluation complete! Results saved to reports/")
            except Exception as exc:
                st.error(f"Evaluation error: {exc}")

    # ── Display results if available ──────────────────────────────────────────
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        st.divider()

        # Metric definitions
        with st.expander("View Metric Definitions"):
            for metric, defn in report["metric_definitions"].items():
                st.markdown(f"**{metric}**: {defn}")

        st.divider()
        st.subheader("Model Comparison")

        ma = report["model_a"]
        mb = report["model_b"]

        c1, c2, c3, c4 = st.columns(4)
        metric_pairs = [
            ("Avg FIS", "average_fis", c1),
            ("Avg TAS", "average_tas", c2),
            ("Avg PESS", "average_pess", c3),
            ("Avg Final Score", "average_final_score", c4),
        ]
        for label, key, col in metric_pairs:
            with col:
                delta = ma[key] - mb[key]
                st.metric(
                    label=f"Model A — {label}",
                    value=f"{ma[key]:.1f}",
                    delta=f"{delta:+.1f} vs B",
                    delta_color="normal",
                )

        # Summary table
        st.divider()
        summary_data = {
            "Metric": ["FIS", "TAS", "PESS", "Final Score"],
            "Model A": [
                ma["average_fis"],
                ma["average_tas"],
                ma["average_pess"],
                ma["average_final_score"],
            ],
            "Model B": [
                mb["average_fis"],
                mb["average_tas"],
                mb["average_pess"],
                mb["average_final_score"],
            ],
        }
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary.set_index("Metric"), use_container_width=True)

        # Per-scenario results
        if results_path.exists():
            st.divider()
            st.subheader("Per-Scenario Results")
            df_results = pd.read_csv(results_path)
            st.dataframe(df_results, use_container_width=True)

            st.subheader("Final Score by Scenario")
            chart_data = df_results.pivot_table(
                index="ScenarioID", columns="Model", values="FinalScore"
            )
            st.bar_chart(chart_data, use_container_width=True)

    else:
        st.info(
            "No evaluation results found yet. "
            "Click **Run Full Evaluation** above to generate them."
        )
