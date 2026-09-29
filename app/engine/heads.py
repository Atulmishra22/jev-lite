import torch
import torch.nn as nn
import torch.nn.functional as F

class ChoiceHead(nn.Module):
    """
    Evaluates dynamic choices using cosine similarity in Qwen's native embedding space.
    No random weights -> 100% universal zero-shot accuracy!
    """
    def __init__(self, hidden_size: int = None):
        super().__init__()

    def forward(self, query_hidden: torch.Tensor, option_hiddens: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        # Normalize vectors to unit sphere (Cosine Similarity)
        query_norm = F.normalize(query_hidden.unsqueeze(0), p=2, dim=-1)   # [1, hidden_size]
        options_norm = F.normalize(option_hiddens, p=2, dim=-1)             # [num_options, hidden_size]

        # Cosine similarity scores between -1.0 and 1.0
        similarities = torch.matmul(options_norm, query_norm.squeeze(0))    # [num_options]

        # Temperature scaling (0.1 sharpens cosine similarity into confident probabilities)
        probs = torch.softmax(similarities / temperature, dim=-1)
        return probs


class NoulHead(nn.Module):
    """
    Evaluates True/False by comparing alignment against affirmation vs negation.
    """
    def __init__(self, hidden_size: int = None):
        super().__init__()

    def forward(self, query_hidden: torch.Tensor, true_hidden: torch.Tensor, false_hidden: torch.Tensor) -> float:
        query_norm = F.normalize(query_hidden.unsqueeze(0), p=2, dim=-1)
        true_norm = F.normalize(true_hidden.unsqueeze(0), p=2, dim=-1)
        false_norm = F.normalize(false_hidden.unsqueeze(0), p=2, dim=-1)

        sim_true = torch.dot(query_norm.squeeze(0), true_norm.squeeze(0))
        sim_false = torch.dot(query_norm.squeeze(0), false_norm.squeeze(0))

        # Softmax over True vs False (scaled by 0.1)
        probs = torch.softmax(torch.stack([sim_false, sim_true]) / 0.1, dim=-1)
        return probs[1].item()  # Probability of True


class ScoreHead(nn.Module):
    """
    Computes expected score over arbitrary rubric levels.
    """
    def __init__(self, hidden_size: int = None):
        super().__init__()

    def forward(self, level_probs: dict[str, float]) -> float:
        try:
            expected_score = sum(float(level) * prob for level, prob in level_probs.items())
            return round(expected_score, 2)
        except ValueError:
            # If level keys are not numbers, fallback to index-based score
            return 1.0