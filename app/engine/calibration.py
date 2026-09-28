import torch

def calculate_choice_confidence(probabilties: dict[str, float] | list[float]) -> float:
    """
    computes calibrated confidence for a choice distribution.
    Uses TypeSafe's peak-spread formula:
    confidence = (K * p_max - 1) / (k -1 )
    """
    if isinstance(probabilties, dict):
        probs = list(probabilties.values())
    else:
        probs = probabilties
        
    count = len(probs)
    if count  <= 1:
        return 1.0

    peak = max(probs)
    raw_confidence = (count * peak - 1.0 ) / ( count - 1.0)

    return max(0.0, min(1.0, float(raw_confidence)))

def calculate_score_confidence(probabilities: dict[str, float] | list[float]) -> float:
    """
    computes confidence for a score distribution.
    A peaked distribution inidicates high confidence, while a flat distribution indicates uncertainty.
    """
    return calculate_choice_confidence(probabilities)

def apply_temperature_scaling(logits: torch.Tensor, temperature: float= 1.0) -> torch.Tensor:
    """
    Applies Platt/ temperature scaling to raw logits.
    T > 1.0 softes overconfident distributions.
    T < 1.0 sharpens distributions.

    """
    if temperature <= 0.0:
        raise ValueError("Temperature must be greater than 0.0")
    return logits / temperature
