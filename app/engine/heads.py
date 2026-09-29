import torch
import torch.nn as nn

class NoulHead(nn.Module):
    """
    Evaluates True/False statements.
    projects hidden_size ->  1 scalar logit -> Sigmoid (0.0 to 1.0 probaility).
    """
    def __init__(self, hidden_size: int):
        super().__init__()        
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states shape : [batch_size, hidden_size]
        logits = self.classifier(hidden_states)  # shape : [batch_size, 1]
        probs = torch.sigmoid(logits)  # shape : [batch_size, 1]
        return probs.squeeze(-1) # [batch_size]

class ScoreHead(nn.Module):
    """
    evaluates numeric/ rubric scores.
    projects hidden_size -> continous scalar value.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        score = self.regressor(hidden_state)
        return score.squeeze(-1)

class ChoiceHead(nn.Module):
    """
    Evaluates dynamic choices.
    projects candidate option representation into matching logits,
    then applies softmax with temperature.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.project = nn.Linear(hidden_size, hidden_size)

    def forward(self, query_hidden: torch.Tensor, option_hiddens: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        query_hidden : [hidden_size] (represtation of state + question)
        option_hiddens : [num_options, hidden_size] (representation of the critearia options)
        returns : [num_options] probability distribution summing to 1.0
        """
        # project query
        projected_query = self.project(query_hidden)  # shape : [hidden_size]

        # dot-product similarity between query and each option 
        logits = torch.matmul( option_hiddens, projected_query) / (query_hidden.shape[-1] ** 0.5)

        # apply softmax with temperature
        probs = torch.softmax(logits / temperature, dim=-1)  # shape : [num_options]
        return probs