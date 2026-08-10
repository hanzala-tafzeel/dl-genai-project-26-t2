"""Smart MCQ Solver — Streamlit web app.

Deploys the from-scratch BiLSTM MCQ scorer trained in
notebooks/dl-23f2004513-notebook-t22026.ipynb (val mAP@3 ~0.99, acc ~0.98,
F1 ~0.98). Inference-only: the trained model and vocabulary are loaded once
from artifacts/ at startup — no training, no wandb, no Kaggle secrets.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mcq_model import (
    OPTION_COLS,
    MCQScorer,
    predict_top3_for_question,
    score_options,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@st.cache_resource(show_spinner="Loading Smart MCQ Solver model...")
def load_model():
    if not ARTIFACTS_DIR.is_dir():
        st.error(
            "Deployment artifacts not found. Run `python scripts/build_artifacts.py "
            "--train train.csv` locally once and push the `artifacts/` folder to GitHub."
        )
        st.stop()

    with open(ARTIFACTS_DIR / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(ARTIFACTS_DIR / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    model = MCQScorer(config["vocab_size"], config["emb_dim"], config["hidden_dim"])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "model.pt", map_location="cpu"))
    model.eval()
    return model, vocab, config


def main():
    st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="wide")

    model, vocab, config = load_model()

    with st.sidebar:
        st.title("🧠 Smart MCQ Solver")
        st.caption("From-scratch BiLSTM MCQ scorer · DL/GenAI Project (T2-2026)")
        st.divider()
        st.markdown("**Model card**")
        st.markdown(
            f"- Architecture: Shared BiLSTM encoder + MLP scorer\n"
            f"- Embedding dim: {config['emb_dim']}\n"
            f"- Hidden dim: {config['hidden_dim']}\n"
            f"- Vocabulary size: {config['vocab_size']:,}\n"
            f"- Max sequence length: {config['max_len']} tokens\n"
            f"- Trained for: {config['epochs']} epochs, seed {config['seed']}"
        )
        st.divider()
        st.markdown("**Validation metrics**")
        st.markdown(
            f"- mAP@3: **{config.get('final_val_map3', 'n/a')}**\n"
            f"- Accuracy (top-1): **{config.get('accuracy', 'n/a')}**\n"
            f"- F1 (macro): **{config.get('f1_macro', 'n/a')}**"
        )
        st.divider()
        st.caption(
            "Given a question prompt and five options, the model scores each "
            "option and returns the ranked top-3 most likely correct answers."
        )

    st.title("Solve an MCQ")
    st.markdown(
        "Enter a question and its five options, then press **Solve MCQ** to get "
        "the model's ranked top-3 prediction (submission format: `A E B`)."
    )

    prompt = st.text_area(
        "Question / Prompt",
        height=110,
        max_chars=800,
        placeholder="e.g. Pick the best possible answer: What is Martin Heidegger's view on the relationship between time and human existence? among the listed options.",
        key="prompt",
    )

    cols = st.columns(5)
    option_inputs = {}
    for col, letter in zip(cols, OPTION_COLS):
        option_inputs[letter] = col.text_area(
            f"Option {letter}",
            height=170,
            max_chars=500,
            key=f"option_{letter}",
        )

    solve = st.button("🚀 Solve MCQ", type="primary", width="stretch")

    if solve:
        with st.spinner("Scoring options..."):
            if not prompt.strip():
                st.warning("Please enter a question prompt.")
                st.stop()
            if all(not (option_inputs[l] or "").strip() for l in OPTION_COLS):
                st.warning("Please fill in at least one option.")
                st.stop()

            try:
                scores = score_options(model, prompt, option_inputs, vocab)
                top3 = predict_top3_for_question(model, prompt, option_inputs, vocab)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        st.success(f"**Top 3 Prediction: `{' '.join(top3)}`**")
        st.write(
            f"**Most likely answer: Option {top3[0]}** "
            f"({scores[top3[0]]:.1%} confidence)"
        )

        chart_df = pd.DataFrame(
            {"Option": list(scores.keys()), "Confidence": list(scores.values())}
        ).set_index("Option")

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.subheader("Option confidence")
            st.bar_chart(chart_df, height=280, color="#2E5090")
        with col_table:
            st.subheader("Scores")
            score_rows = pd.DataFrame(
                [
                    {
                        "Option": letter,
                        "Score": f"{prob:.4f}",
                        "Confidence": f"{prob:.1%}",
                        "Rank": rank + 1,
                    }
                    for rank, (letter, prob) in enumerate(
                        sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    )
                ]
            )
            st.dataframe(score_rows, hide_index=True, width="stretch")

        st.caption(
            "Scores come from the notebook's exact pipeline: the same cleaning, "
            "40-token vocabulary encoding, shared BiLSTM encoder and classifier."
        )


if __name__ == "__main__":
    main()