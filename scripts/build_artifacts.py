"""One-time artifact builder for the Smart MCQ Solver deployment.

Reproduces the training performed in
notebooks/dl-23f2004513-notebook-t22026.ipynb (from-scratch BiLSTM,
seed 42, 8 epochs, Adam 1e-3, batch 64, BCEWithLogitsLoss) and saves the
deployment artifacts consumed by app.py:

  artifacts/vocab.json    word -> token id (vocabulary built from train.csv)
  artifacts/model.pt      trained MCQScorer state_dict
  artifacts/config.json   hyperparameters + final validation metrics

Run locally ONCE (never on Streamlit Cloud):
    python scripts/build_artifacts.py --train train.csv --out artifacts

No wandb, no Kaggle secrets, no GPU required.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcq_model import (
    OPTION_COLS,
    MAX_LEN,
    MCQScorer,
    clean_text,
    encode,
    predict_top3_for_question,
    tokenize,
)

SEED = 42
MIN_FREQ = 2
EMB_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 8
VAL_FRAC = 0.15


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def average_precision_at_3(true_label, predicted_ranking):
    top3 = predicted_ranking[:3]
    if true_label in top3:
        return 1.0 / (top3.index(true_label) + 1)
    return 0.0


def mean_average_precision_at_3(true_labels, predicted_rankings):
    scores = [average_precision_at_3(t, p) for t, p in zip(true_labels, predicted_rankings)]
    return float(np.mean(scores))


def top1_accuracy_and_f1(true_labels, predicted_rankings):
    pred_top1 = [ranking[0] for ranking in predicted_rankings]
    acc = np.mean([t == p for t, p in zip(true_labels, pred_top1)])
    macro_f1 = f1_macro(true_labels, pred_top1)
    return acc, macro_f1


def f1_macro(true_labels, predicted_labels):
    labels = OPTION_COLS
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == label and p != label)
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


class MCQDataset(Dataset):
    """Each item = (prompt_ids, option_ids, label) for one (question, option) pair."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        _, prompt_ids, option_ids, label = self.rows[idx]
        return (
            torch.tensor(prompt_ids, dtype=torch.long),
            torch.tensor(option_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def build_vocab(train_df):
    counter = Counter()
    for _, row in train_df.iterrows():
        counter.update(tokenize(clean_text(row["prompt"])))
        for col in OPTION_COLS:
            counter.update(tokenize(clean_text(row[col])))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= MIN_FREQ:
            vocab[word] = len(vocab)
    return vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="train.csv", help="Path to train.csv")
    parser.add_argument("--out", default="artifacts", help="Output directory for artifacts")
    args = parser.parse_args()

    set_seed(SEED)

    train_df = pd.read_csv(args.train)
    print(f"Loaded {args.train}: {train_df.shape}")

    vocab = build_vocab(train_df)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    rows = []
    for _, row in train_df.iterrows():
        prompt_ids = encode(row["prompt"], vocab)
        for col in OPTION_COLS:
            option_ids = encode(row[col], vocab)
            label = 1 if row["answer"] == col else 0
            rows.append((row["id"], prompt_ids, option_ids, label))
    print(f"Total (prompt, option) training rows: {len(rows)}")

    qids = train_df["id"].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(qids)
    split_idx = int((1 - VAL_FRAC) * len(qids))
    train_qids, val_qids = set(qids[:split_idx]), set(qids[split_idx:])

    train_rows = [r for r in rows if r[0] in train_qids]
    val_rows = [r for r in rows if r[0] in val_qids]
    print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)}")

    train_loader = DataLoader(MCQDataset(train_rows), batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MCQScorer(vocab_size, EMB_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    val_df = train_df[train_df["id"].isin(val_qids)].reset_index(drop=True)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for prompt_ids, option_ids, labels in train_loader:
            prompt_ids, option_ids, labels = (
                prompt_ids.to(device),
                option_ids.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            logits = model(prompt_ids, option_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_rows)
        val_map3 = evaluate_map3(model, val_df, vocab, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_map3": val_map3})
        print(f"Epoch {epoch:>2}/{EPOCHS} | train_loss={train_loss:.4f} | val_mAP@3={val_map3:.4f}")

    val_predicted_rankings = [
        predict_top3_for_question(
            model,
            row["prompt"],
            {c: row[c] for c in OPTION_COLS},
            vocab,
            device=device,
        )
        for _, row in val_df.iterrows()
    ]
    final_val_map3 = mean_average_precision_at_3(val_df["answer"].tolist(), val_predicted_rankings)
    acc, f1 = top1_accuracy_and_f1(val_df["answer"].tolist(), val_predicted_rankings)
    print(f"BiLSTM -> val_mAP@3={final_val_map3:.4f} | accuracy={acc:.4f} | f1_macro={f1:.4f}")

    save_artifacts(args.out, model, vocab, {
        "vocab_size": vocab_size,
        "emb_dim": EMB_DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_len": MAX_LEN,
        "min_freq": MIN_FREQ,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "final_val_map3": final_val_map3,
        "accuracy": acc,
        "f1_macro": f1,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "history": history,
    })


def evaluate_map3(model, df_subset, vocab, device):
    true_labels, predicted_rankings = [], []
    for _, row in df_subset.iterrows():
        option_texts = {col: row[col] for col in OPTION_COLS}
        pred = predict_top3_for_question(model, row["prompt"], option_texts, vocab, device=device)
        true_labels.append(row["answer"])
        predicted_rankings.append(pred)
    return mean_average_precision_at_3(true_labels, predicted_rankings)


def save_artifacts(out_dir, model, vocab, config):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=1)
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved artifacts -> {out / 'vocab.json'}, {out / 'model.pt'}, {out / 'config.json'}")


if __name__ == "__main__":
    main()