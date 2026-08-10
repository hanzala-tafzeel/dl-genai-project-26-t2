# Smart MCQ Solver — DL/GenAI Project (T2-2026)

A deep learning project that answers multiple-choice questions (options A–E) by learning to
rank the given options against the question prompt, trained from scratch with a BiLSTM
scorer and benchmarked against pretrained Sentence-Transformer and fine-tuned DistilBERT
baselines. Experiments are tracked with Weights & Biases.

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Preprocessing](#4-preprocessing)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Model — Built From Scratch](#6-model--built-from-scratch)
7. [Evaluation Metric — mAP@3](#7-evaluation-metric--map3)
8. [Training](#8-training)
9. [Results](#9-results)
10. [Inference & Submission](#10-inference--submission)
11. [Bonus — Pretrained Model Baselines](#11-bonus--pretrained-model-baselines)
12. [Comparison of Approaches](#12-comparison-of-approaches)
13. [Project Structure](#13-project-structure)
14. [Setup & Installation](#14-setup--installation)
15. [How to Run](#15-how-to-run)
16. [Experiment Tracking](#16-experiment-tracking)

---

## 1. Problem Statement
Given a question `prompt` and five candidate answers (`A`–`E`), predict the top-3 most
likely correct options, ranked by confidence. Performance is scored with **mAP@3**
(mean Average Precision @ 3).

## 2. Dataset
| File | Description |
|---|---|
| `train.csv` | `id, prompt, A, B, C, D, E, answer` — labeled training questions |
| `test.csv` | `id, prompt, A, B, C, D, E` — unlabeled evaluation questions |
| `sample_submission.csv` | `ID, Prediction` — expected submission format, e.g. `A B C` |

## 3. Exploratory Data Analysis
- Label (correct-answer) distribution across A–E to check class balance
- Prompt and option length distributions
- Text overlap/similarity checks between prompt and options

## 4. Preprocessing
- Text cleaning (lowercasing, punctuation/whitespace normalization)
- Vocabulary built from the training corpus
- Encoding function to convert cleaned text into token-index sequences

## 5. Dataset Preparation
- Training rows reframed as **binary classification pairs**: (prompt, option) → is this
  option correct?
- Custom PyTorch `Dataset` class to serve `(prompt, option)` pairs and labels

## 6. Model — Built From Scratch
- **Shared BiLSTM Encoder** — encodes the prompt and each option through a common
  embedding + bidirectional LSTM
- **MCQ Scorer** head — combines prompt/option representations and outputs a relevance
  score per option
- At inference, the 5 option scores per question are ranked to produce the top-3 prediction

## 7. Evaluation Metric — mAP@3
Custom implementation of mean Average Precision @ 3, used both for validation during
training and for scoring the final held-out validation split.

## 8. Training
- Hyperparameters (epochs, learning rate, optimizer) and loss function defined
- Training loop with per-epoch loss + validation mAP@3, logged to Weights & Biases
- Training/validation curves plotted after training

## 9. Results
- Final validation mAP@3 reported
- Sanity checks run to rule out dataset shortcuts inflating the score (e.g. positional or
  length bias in the correct option)

## 10. Inference & Submission
- Trained model run on `test.csv`
- Top-3 ranked options per question written out in `sample_submission.csv` format

## 11. Bonus — Pretrained Model Baselines
- **Sentence-Transformer** (no fine-tuning) — embedding similarity between prompt and
  options, used as a zero-shot baseline
- **Fine-tuned DistilBERT** — pretrained transformer fine-tuned on the same training data,
  evaluated with the same mAP@3 metric

## 12. Comparison of Approaches
Side-by-side comparison of the from-scratch BiLSTM model against the pretrained
Sentence-Transformer and fine-tuned DistilBERT baselines on validation mAP@3.

## 13. Project Structure
```
.
├── app.py                                  # Streamlit web app (inference only)
├── src/
│   └── mcq_model.py                        # notebook-exact model + preprocessing (deployment)
├── scripts/
│   └── build_artifacts.py                  # one-time artifact builder (trains + saves model/vocab)
├── artifacts/                              # trained model.pt + vocab.json + config.json (committed)
├── notebooks/
│   └── dl-23f2004513-notebook-t22026.ipynb   # end-to-end notebook (EDA → model → submission)
├── requirements.txt
└── README.md
```

## 14. Setup & Installation
```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

## 15. How to Run
1. Place `train.csv`, `test.csv`, and `sample_submission.csv` in the project root (or
   update the paths in the notebook).
2. Open `notebooks/dl-23f2004513-notebook-t22026.ipynb`.
3. Run cells in order: imports → EDA → preprocessing → model training → evaluation →
   inference → submission generation.
4. Submit the generated predictions file to the Kaggle competition leaderboard.

## 16. Web App Deployment (Streamlit Community Cloud)

The repo ships with a ready-to-deploy Streamlit app (`app.py`) that runs the trained
from-scratch BiLSTM model **without retraining and without any Kaggle/W&B secrets**.

### Local run
```bash
streamlit run app.py
```

### (Re)build the deployment artifacts
The trained model and vocabulary are stored in `artifacts/` (`model.pt`, `vocab.json`,
`config.json`) and are committed to the repo — the app loads them once at startup.
To regenerate them from `train.csv` (same seed/epochs/hyperparameters as the notebook):

```bash
python scripts/build_artifacts.py --train train.csv --out artifacts
```
Running this is only needed if the model is retrained; `train.csv` itself is never committed.

### Deploy to Streamlit Community Cloud
1. Push this repo to GitHub (set the working directory to the repo root).
2. Go to https://share.streamlit.io (or the Community Cloud dashboard).
3. Click **Create app** → **Deploy from GitHub repo**.
4. Select the repository, branch `main`, and main file path `app.py`.
5. (Optional) Advanced settings → Python version 3.11 or newer. No secrets needed.
6. Click **Deploy**. Cold start takes ~1–2 minutes (installs torch etc.); the model then
   loads from `artifacts/` and the app is ready.

## 17. Experiment Tracking
Training runs, losses, and validation mAP@3 are logged to **Weights & Biases** (`wandb`).
Set your `WANDB_API_KEY` before running the notebook, or log in interactively when
prompted at the WandB setup cell.
