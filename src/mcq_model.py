"""Smart MCQ Solver — deployment model.

Verbatim port of the inference pipeline from
notebooks/dl-23f2004513-notebook-t22026.ipynb (from-scratch BiLSTM MCQ scorer):

  clean_text -> tokenize -> encode (vocab lookup, MAX_LEN=40)
  -> SharedEncoder (Embedding + BiLSTM, mean-pooled) -> MCQScorer
  -> sigmoid score per option -> rank -> top-3.

No wandb, no kaggle secrets, no training code. The model and vocabulary are
loaded once from the artifacts/ directory (see scripts/build_artifacts.py).
"""

import re

import pandas as pd
import torch
import torch.nn as nn

OPTION_COLS = ["A", "B", "C", "D", "E"]
MAX_LEN = 40
PAD_ID = 0
UNK_ID = 1


# ---------------------------------------------------------------------------
# Preprocessing (Section 3 of the notebook)
# ---------------------------------------------------------------------------

def clean_text(text):
    """Lowercase, strip punctuation/digits-noise, collapse whitespace.
    Returns "" for NaN instead of crashing -- some option cells could be empty."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()


def encode(text, vocab, max_len=MAX_LEN):
    """Text -> fixed-length list of token ids using the deployed vocab."""
    tokens = tokenize(clean_text(text))
    ids = [vocab.get(t, UNK_ID) for t in tokens[:max_len]]
    ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids


# ---------------------------------------------------------------------------
# Model (Section 5 of the notebook)
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """Embeds + BiLSTM-encodes a token sequence into one fixed-size vector
    (mean-pooled over non-pad tokens)."""

    def __init__(self, vocab_size, emb_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, token_ids):
        pad_mask = (token_ids != PAD_ID).unsqueeze(-1).float()
        embedded = self.embedding(token_ids)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out * pad_mask
        pooled = lstm_out.sum(1) / pad_mask.sum(1).clamp(min=1)
        return pooled


class MCQScorer(nn.Module):
    """Scores how well an option answers a prompt: shared encoder for prompt
    and option, combined with [p, o, |p-o|, p*o], then an MLP classifier."""

    def __init__(self, vocab_size, emb_dim=64, hidden_dim=64):
        super().__init__()
        self.encoder = SharedEncoder(vocab_size, emb_dim, hidden_dim)
        combined_dim = hidden_dim * 2 * 4
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, prompt_ids, option_ids):
        p_vec = self.encoder(prompt_ids)
        o_vec = self.encoder(option_ids)
        combined = torch.cat([p_vec, o_vec, torch.abs(p_vec - o_vec), p_vec * o_vec], dim=1)
        return self.classifier(combined).squeeze(-1)


def build_model(vocab_size, emb_dim=64, hidden_dim=64):
    return MCQScorer(vocab_size, emb_dim, hidden_dim)


# ---------------------------------------------------------------------------
# Inference (Section 7 of the notebook)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_options(model, prompt_text, option_texts, vocab, max_len=MAX_LEN, device="cpu"):
    """Returns {letter: sigmoid score} for every option."""
    model.eval()
    prompt_ids = torch.tensor([encode(prompt_text, vocab, max_len)], device=device)
    scores = {}
    for letter, text in option_texts.items():
        option_ids = torch.tensor([encode(text, vocab, max_len)], device=device)
        logit = model(prompt_ids, option_ids)
        scores[letter] = torch.sigmoid(logit).item()
    return scores


def predict_top3_for_question(model, prompt_text, option_texts, vocab, max_len=MAX_LEN, device="cpu"):
    """option_texts: dict like {'A': text, ...}. Returns top-3 option letters, ranked."""
    scores = score_options(model, prompt_text, option_texts, vocab, max_len, device)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [letter for letter, _ in ranked[:3]]