import re
from typing import List, Dict

# ---------- Ordered bands for "near-miss" scoring ----------
ORDERED_BANDS = {
    "returns":    ["bad", "neutral", "good"],
    "volatility": ["low", "medium", "high"],
}

# ---------- XML helper ----------
def _parse_prediction(completion: str, tag: str) -> str | None:
    """
    Extracts <tag>value</tag> from the model's XML-like output.
    Returns the value in lowercase, or None if the tag is missing.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, completion, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip().lower() if match else None

# ---------- Scoring helper ----------
def _calculate_score(predicted: str | None,
                     actual: str,
                     band_type: str) -> float:
    """
    Reward schedule:
      • +1   exact match
      • +0.1 one-step "near miss"
      • −1   wrong band
      • −2   missing / hallucinated value
    """
    if predicted is None or actual is None:
        return -2.0

    if predicted not in ORDERED_BANDS[band_type]:
        return -2.0

    if predicted == actual:
        return 1.0

    try:
        p_idx = ORDERED_BANDS[band_type].index(predicted)
        a_idx = ORDERED_BANDS[band_type].index(actual)
        if abs(p_idx - a_idx) == 1:
            return 0.1
    except ValueError:
        return -2.0  # safety net

    return -1.0

# ---------- Main reward function ----------
def financial_reward_function(
    *,
    prompts: List[str],
    completions: List[str],
    ground_truth_returns_1y: List[str],
    ground_truth_volatility_1y: List[str],
    ground_truth_returns_5y: List[str],
    ground_truth_volatility_5y: List[str],
    **kwargs,
) -> List[float]:
    """
    Computes a scalar reward for each (prompt, completion) pair.

    The trainer passes every dataset column as **kwargs**; we declare the ones
    we need explicitly for clarity and IDE autocompletion.
    """
    batch_rewards: List[float] = []

    for i, completion in enumerate(completions):
        # 1. Parse predictions from the generated text
        pred_ret_1y   = _parse_prediction(completion, "returns_1y")
        pred_vol_1y   = _parse_prediction(completion, "volatility_1y")
        pred_ret_5y   = _parse_prediction(completion, "returns_5y")
        pred_vol_5y   = _parse_prediction(completion, "volatility_5y")

        # 2. Compare with ground truth for this sample
        score_ret_1y = _calculate_score(pred_ret_1y,
                                        ground_truth_returns_1y[i], "returns")
        score_vol_1y = _calculate_score(pred_vol_1y,
                                        ground_truth_volatility_1y[i], "volatility")
        score_ret_5y = _calculate_score(pred_ret_5y,
                                        ground_truth_returns_5y[i], "returns")
        score_vol_5y = _calculate_score(pred_vol_5y,
                                        ground_truth_volatility_5y[i], "volatility")

        # 3. Aggregate
        total_reward = (
            score_ret_1y + score_vol_1y + score_ret_5y + score_vol_5y
        )
        batch_rewards.append(total_reward)

    return batch_rewards
