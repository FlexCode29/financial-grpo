# ───────────────────────────────────────────────────────────────────────────────
#  src/reward.py            complete module with two separate reward functions
# ───────────────────────────────────────────────────────────────────────────────
"""
Reward functions for the RiskClassifier task.

There are two independent rewards:

1. risk_format_reward_function
   • +1 if the completion contains a valid <risk_level> tag whose value is
     one of the four accepted bands.
   • -1 otherwise.

2. risk_correctness_reward_function
   • +1   exact band match
   • +0.1 adjacent band ("near miss")
   • -1   band two or more steps away
   • -2   missing tag or invalid value
"""

from __future__ import annotations
import re
from typing import List

# ---------------------------------------------------------------------------
#  Ordered bands (for both rewards)
# ---------------------------------------------------------------------------
ORDERED_BANDS = ["low risk", "moderate risk", "high risk", "very high risk"]

# ---------------------------------------------------------------------------
#  Regex helpers
# ---------------------------------------------------------------------------
TAG_PATTERN = re.compile(
    r"<risk_level>(.*?)</risk_level>",
    re.IGNORECASE | re.DOTALL,
)

def _parse_prediction(completion: str) -> str | None:
    """
    Returns the text between <risk_level> and </risk_level>, stripped and
    converted to lower case. If the tag is not found, returns None.
    """
    m = TAG_PATTERN.search(completion)
    return m.group(1).strip().lower() if m else None

# ---------------------------------------------------------------------------
#  Format reward
# ---------------------------------------------------------------------------
def risk_format_reward_function(
    *,
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    +1 if a valid risk_level tag with an accepted band exists, else -1.
    """
    rewards: List[float] = []
    for comp in completions:
        pred = _parse_prediction(comp)
        if pred in ORDERED_BANDS:
            rewards.append(0.1)
        else:
            rewards.append(-0.1)
    return rewards

# ---------------------------------------------------------------------------
#  Correctness helpers
# ---------------------------------------------------------------------------
def _band_distance(predicted: str, actual: str) -> int:
    """Absolute index distance between predicted and actual band."""
    return abs(ORDERED_BANDS.index(predicted) - ORDERED_BANDS.index(actual))

def _calculate_correctness_score(predicted: str | None, actual: str) -> float:
    if predicted is None or predicted not in ORDERED_BANDS:
        return -2.0
    if predicted == actual:
        return 1.0
    if _band_distance(predicted, actual) == 1:
        return 0.1
    return -1.0

# ---------------------------------------------------------------------------
#  Correctness reward
# ---------------------------------------------------------------------------
def risk_correctness_reward_function(
    *,
    prompts: List[str],
    completions: List[str],
    ground_truth_risk_band: List[str],
    **kwargs,
) -> List[float]:
    """
    Computes correctness reward for each completion.
    """
    rewards: List[float] = []
    for comp, truth in zip(completions, ground_truth_risk_band):
        pred = _parse_prediction(comp)
        rewards.append(_calculate_correctness_score(pred, truth))
    return rewards
