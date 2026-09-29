import json
from typing import Any, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import settings
from app.engine.calibration import (
    calculate_choice_confidence,
    calculate_score_confidence,
)
from app.schemas.request import SystemOneRequest, ChoiceQuestion, ScoreQuestion, NoulQuestion
from app.schemas.response import SystemOneResponse, ChoiceAnswer, ScoreAnswer, NoulAnswer, Answer


class SystemOneEngine(nn.Module):
    def __init__(self, model_name: str = settings.model_name, device: str = settings.device):
        super().__init__()
        self.device = torch.device(device)
        self.model_version = settings.model_version

        # 1. Load Tokenizer & Causal LM (to access single-pass vocabulary logits)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        self.backbone.to(device=self.device, dtype=self.backbone.dtype)
        self.backbone.eval()

    def _format_state(self, state: Union[str, dict[str, Any], list[Any]]) -> str:
        if isinstance(state, str):
            return state
        return json.dumps(state, indent=2)

    def _format_chat(self, role: str, content: str, add_generation_prompt: bool = False) -> str:
        messages = [{"role": role, "content": content}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

    def _get_first_token_id(self, word: str) -> int:
        """Helper to get the exact token ID for a candidate word."""
        # Prepend a space because in sentence context words are preceded by space
        token_ids = self.tokenizer.encode(" " + word.strip(), add_special_tokens=False)
        return token_ids[0] if token_ids else self.tokenizer.encode(word.strip(), add_special_tokens=False)[0]

    @torch.no_grad()
    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        # Step 1: Ingest State ONCE to get KV-cache
        state_text = f"State Context:\n{self._format_state(request.state)}"
        formatted_state = self._format_chat(role="system", content=state_text, add_generation_prompt=False)
        state_inputs = self.tokenizer(formatted_state, return_tensors="pt").to(self.device)

        state_outputs = self.backbone(**state_inputs, use_cache=True)
        state_kv_cache = state_outputs.past_key_values

        answers: dict[str, Answer] = {}

        # Step 2: Evaluate each question against the cached state in 1 single forward pass
        for q_id, question in request.questions.items():
            if isinstance(question, NoulQuestion):
                # Format truth verification prompt
                q_content = f"Statement: {question.statement}\nIs this statement true? (Answer True or False):"
                q_text = self._format_chat(role="user", content=q_content, add_generation_prompt=True)
                q_inputs = self.tokenizer(q_text, return_tensors="pt").to(self.device)

                # Single forward pass!
                q_out = self.backbone(**q_inputs, past_key_values=state_kv_cache)
                last_token_logits = q_out.logits[0, -1, :]

                # Read logits for candidate words "True" vs "False"
                true_id = self._get_first_token_id("True")
                false_id = self._get_first_token_id("False")
                
                tf_logits = torch.tensor([last_token_logits[false_id].item(), last_token_logits[true_id].item()])
                tf_probs = torch.softmax(tf_logits, dim=-1)
                true_prob = tf_probs[1].item()

                answers[q_id] = NoulAnswer(noul=round(true_prob, 4))

            elif isinstance(question, ChoiceQuestion):
                option_names = list(question.criteria.keys())
                options_str = "\n".join([f"- {k}: {v}" for k, v in question.criteria.items()])
                q_content = f"Question: {question.instructions}\nOptions:\n{options_str}\nSelected Option:"
                q_text = self._format_chat(role="user", content=q_content, add_generation_prompt=True)
                q_inputs = self.tokenizer(q_text, return_tensors="pt").to(self.device)

                # Single forward pass!
                q_out = self.backbone(**q_inputs, past_key_values=state_kv_cache)
                last_token_logits = q_out.logits[0, -1, :]

                # Read logits for each candidate option key
                cand_ids = [self._get_first_token_id(name) for name in option_names]
                cand_logits = torch.tensor([last_token_logits[cid].item() for cid in cand_ids])

                # Softmax over only the candidate options
                probs = torch.softmax(cand_logits, dim=-1).tolist()
                prob_dict = {name: round(p, 4) for name, p in zip(option_names, probs)}
                best_choice = max(prob_dict, key=prob_dict.get)
                confidence = calculate_choice_confidence(prob_dict)

                answers[q_id] = ChoiceAnswer(
                    choice=best_choice,
                    probabilities=prob_dict,
                    confidence=round(confidence, 4)
                )

            elif isinstance(question, ScoreQuestion):
                level_names = list(question.levels.keys())
                levels_str = "\n".join([f"- Level {k}: {v}" for k, v in question.levels.items()])
                q_content = f"Evaluation: {question.instructions}\nRubric:\n{levels_str}\nScore Level:"
                q_text = self._format_chat(role="user", content=q_content, add_generation_prompt=True)
                q_inputs = self.tokenizer(q_text, return_tensors="pt").to(self.device)

                # Single forward pass!
                q_out = self.backbone(**q_inputs, past_key_values=state_kv_cache)
                last_token_logits = q_out.logits[0, -1, :]

                # Read logits for each level (e.g. "1", "2", "3")
                level_ids = [self._get_first_token_id(name) for name in level_names]
                level_logits = torch.tensor([last_token_logits[lid].item() for lid in level_ids])

                probs = torch.softmax(level_logits, dim=-1).tolist()
                prob_dict = {name: round(p, 4) for name, p in zip(level_names, probs)}
                confidence = calculate_score_confidence(prob_dict)

                # Calculate expected value score
                try:
                    expected_score = sum(float(k) * p for k, p in prob_dict.items())
                except ValueError:
                    expected_score = 1.0

                answers[q_id] = ScoreAnswer(
                    score=round(expected_score, 2),
                    probabilities=prob_dict,
                    confidence=round(confidence, 4)
                )

        return SystemOneResponse(model=self.model_version, answers=answers)