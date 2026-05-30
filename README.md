# Bigram Language Models

A study of character-level **bigram language models** that generate new,
name-like words — built while following Andrej Karpathy's
[*makemore*](https://github.com/karpathy/makemore) series.

The project walks through the same idea from two angles and shows that they
arrive at the same answer:

1. **A statistical model** that counts how often each pair of adjacent
   characters occurs and normalises those counts into probabilities.
2. **A single-layer neural network** that learns an equivalent probability
   table through gradient-based optimisation.

Both are scored with the same metric — the average **negative log-likelihood
(NLL)** — so the two approaches can be compared directly.

---

## 📄 Report

A polished, multi-page PDF write-up of the whole project lives in
[`report/BigramModels_Report.pdf`](report/BigramModels_Report.pdf). It is
generated programmatically from the real model, so every figure and number in
it is reproducible.

It covers:

- **Project overview** and the end-to-end pipeline
- **The 27×27 bigram count matrix** as a heatmap
- **Reading the statistics** — which letters tend to start and end names
- **Sampling** brand-new names from the model
- **Evaluation** with negative log-likelihood and Laplace (+1) smoothing
- **The neural-network view** — one-hot encoding, weights, softmax, and the
  initial loss

Regenerate it at any time with:

```bash
python generate_report.py
# -> report/BigramModels_Report.pdf
```

---

## 🧠 What the model does

Every name is wrapped with a special boundary token `.` (e.g. `.emma.`). The
model then learns `P(next character | current character)` for all 27 tokens
(`a`–`z` plus `.`).

- **Counting:** build a `27 × 27` matrix `N`, where `N[i, j]` is how often
  character `j` follows character `i`. Normalise each row (with `+1`
  smoothing) to get the probability matrix `P`.
- **Sampling:** start at `.`, repeatedly draw the next character from `P`
  until `.` is drawn again.
- **Neural net:** one-hot encode the input character, multiply by a learnable
  weight matrix `W (27 × 27)`, then apply `exp` + normalisation (softmax) to
  get the same kind of probability distribution. The negative log-likelihood
  is minimised with gradient descent.

### Headline numbers (on the ~32k makemore names dataset)

| Metric | Value |
| --- | --- |
| Names in dataset | 32,033 |
| Token vocabulary | 27 (`a`–`z` + `.`) |
| Bigram transitions counted | 228,146 |
| Average NLL (counting model) | **2.4544** |
| Initial NN loss (untrained) | 3.7693 |

---

## 🚀 Getting started

```bash
# Install dependencies
pip install torch matplotlib numpy

# Run the model walk-through (downloads names.txt on first run)
python make_more.py

# Generate the PDF report
python generate_report.py
```

> The dataset (`data/names.txt`) is downloaded automatically on first run and
> is git-ignored.

---

## 📁 Repository layout

| Path | Description |
| --- | --- |
| `make_more.py` | The step-by-step model walk-through (counting → sampling → NLL → neural net). |
| `make_more.ipynb` | The original Colab notebook version. |
| `generate_report.py` | Reproduces the model and renders the PDF report. |
| `report/BigramModels_Report.pdf` | The generated visual report. |

---

## 📚 Reference

Andrej Karpathy — *The spelled-out intro to language modeling: building
makemore*. This repository contains the lecture-along code and my notes on it.
