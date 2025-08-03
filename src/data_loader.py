# ───────────────────────────────────────────────────────────────────────────────
#  src/data_loader.py          (replaces your previous financial loader)
# ───────────────────────────────────────────────────────────────────────────────
"""
Utility for loading and preparing the theeseus-ai/RiskClassifier dataset.

Each example is turned into a single prompt that concatenates the scenario
`context` and `query`, and provides the four multiple-choice answers so that the
model can reason step-by-step before selecting one of the risk-level bands:
    • low risk
    • moderate risk
    • high risk
    • very high risk
The ground-truth label is stored in column `ground_truth_risk_band`.
"""

from __future__ import annotations
import random
from typing import List, Dict

import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
#  Dataset constants
# ---------------------------------------------------------------------------
RISK_DATASET_REPO = "theeseus-ai/RiskClassifier"
ORDERED_BANDS: List[str] = ["low risk", "moderate risk",
                            "high risk", "very high risk"]

def _score_to_band(score: int) -> str:
    """
    Maps the numeric `risk_score` (0-100) onto one of the four bands.
      0-24   → low risk
      25-49  → moderate risk
      50-74  → high risk
      75-100 → very high risk
    """
    if score < 25:
        return "low risk"
    if score < 50:
        return "moderate risk"
    if score < 75:
        return "high risk"
    return "very high risk"

def _build_prompt(row: Dict) -> str:
    """
    Formats a single example into a text prompt.
    """
    answers = ", ".join(row["answers"])
    return (
        f"{row['context']}\n\n"
        f"Question: {row['query']}\n"
        f"Options: {answers}\n"
        f"Select the most appropriate risk level."
    )

# ---------------------------------------------------------------------------
#  Public loader
# ---------------------------------------------------------------------------
def load_risk_dataset(
    split: str = "train",
    test_size: float = 0.2,
    k_folds: int = 1,
    fold_index: int = 0,
    seed: int = 42,
) -> Dataset:
    """
    Returns a 🤗 `datasets.Dataset` ready for training or evaluation.

    Columns emitted:
        • prompt                 (str)  – the formatted prompt
        • ground_truth_risk_band (str)  – one of the four ordered bands
    """
    # 1. Load the parquet file from the hub
    ds = load_dataset(RISK_DATASET_REPO, split="train")

    # 2. Derive band label and build prompt
    ds = ds.map(
        lambda r: {
            "ground_truth_risk_band": _score_to_band(int(r["risk_score"])),
            "prompt": _build_prompt(r),
        },
        remove_columns=ds.column_names,
    )

    # 3. Shuffle (for determinism in CV)
    ds = ds.shuffle(seed=seed)

    # 4. Classic train/test split or K-fold CV
    if k_folds > 1:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        train_idx, test_idx = list(kf.split(ds))[fold_index]
        train_ds = ds.select(train_idx)
        test_ds  = ds.select(test_idx)
        return train_ds if split == "train" else test_ds

    split_ds = ds.train_test_split(test_size=test_size, seed=seed)
    return split_ds[split]
