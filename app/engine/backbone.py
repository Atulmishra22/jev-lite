import json
from typing import Any, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

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

        # batched casual inferencce : pad on the LEFT
        self.tokenizer.padding_side = "left"

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

    def _expand_kv_cache(self, kv_cache, batch_size: int):
        """Expands the state KV-cache to match the question batch size."""
        if batch_size == 1:
            return kv_cache

        import copy

        # Use native DynamicCache method designed specifically for prompt/state caching
        if hasattr(kv_cache, "batch_repeat_interleave"):
            expanded = copy.deepcopy(kv_cache)
            expanded.batch_repeat_interleave(batch_size)
            return expanded

        return tuple(
            (k.repeat(batch_size, 1, 1, 1), v.repeat(batch_size, 1, 1, 1))
            for k, v in kv_cache
        )

    def _score_candidates(
        self,
        prefix_prompt: str,
        candidates: dict[str, str],
        state_kv_cache,
        state_seq_len: int,
        temperature: float = 0.2,
        length_penalty: float = 0.7,
    ) -> dict[str, float]:
        """
        Evaluates candidate full option sequences in ONE parallel forward pass
        using the state's cached KV states.
        """
        cand_keys = list(candidates.keys())
        prefix_text = self._format_chat(role="user", content=prefix_prompt, add_generation_prompt=True)
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        prefix_len = len(prefix_ids)

        full_sequences = []
        target_spans = []
        for k in cand_keys:
            cand_text = f" {candidates[k].strip()}"
            cand_ids = self.tokenizer.encode(cand_text, add_special_tokens=False)
            seq = prefix_ids + cand_ids
            full_sequences.append(torch.tensor(seq, dtype=torch.long))
            target_spans.append((prefix_len, len(seq)))

        num_cands = len(cand_keys)
        from torch.nn.utils.rnn import pad_sequence
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded_inputs = pad_sequence(full_sequences, batch_first=True, padding_value=pad_id).to(self.device)

        inputs_mask = (padded_inputs != pad_id).long()
        state_mask = torch.ones((num_cands, state_seq_len), dtype=torch.long, device=self.device)
        full_mask = torch.cat([state_mask, inputs_mask], dim=1)

        batched_kv = self._expand_kv_cache(state_kv_cache, num_cands)

        outputs = self.backbone(
            input_ids=padded_inputs,
            attention_mask=full_mask,
            past_key_values=batched_kv,
            use_cache=False
        )

        log_probs = torch.log_softmax(outputs.logits, dim=-1)

        scores = []
        for i, (start_idx, end_idx) in enumerate(target_spans):
            # In causal LM, token at position pos is predicted by logits at pos - 1
            pred_logits = log_probs[i, start_idx - 1 : end_idx - 1, :]
            target_tokens = padded_inputs[i, start_idx : end_idx]
            cand_token_log_probs = pred_logits.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
            token_count = float(end_idx - start_idx)
            cand_score = cand_token_log_probs.sum() / (token_count ** length_penalty)
            scores.append(cand_score)

        score_tensor = torch.stack(scores)
        calibrated_probs = torch.softmax(score_tensor / temperature, dim=-1).tolist()
        return {k: round(p, 4) for k, p in zip(cand_keys, calibrated_probs)}

    @torch.no_grad()
    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        # Step 1: Ingest State ONCE to get KV-cache
        state_text = f"State Context:\n{self._format_state(request.state)}"
        formatted_state = self._format_chat(role="system", content=state_text, add_generation_prompt=False)
        state_inputs = self.tokenizer(formatted_state, return_tensors="pt").to(self.device)
        state_seq_len = state_inputs.input_ids.shape[1]

        state_outputs = self.backbone(**state_inputs, use_cache=True)
        state_kv_cache = state_outputs.past_key_values

        answers: dict[str, Answer] = {}

        for qid, q in request.questions.items():
            if isinstance(q, NoulQuestion):
                prefix = f"Statement: {q.statement}\nIs this statement true or false?"
                candidates = {
                    "true": "Yes, this statement is accurate and true.",
                    "false": "No, this statement is incorrect and false."
                }
                prob_dict = self._score_candidates(
                    prefix_prompt=prefix,
                    candidates=candidates,
                    state_kv_cache=state_kv_cache,
                    state_seq_len=state_seq_len,
                    temperature=settings.default_temperature
                )
                answers[qid] = NoulAnswer(noul=round(prob_dict["true"], 4))

            elif isinstance(q, ChoiceQuestion):
                prefix = f"Question: {q.instructions}\nSelected Option:"
                candidates = {k: f"{k}: {v}" for k, v in q.criteria.items()}
                prob_dict = self._score_candidates(
                    prefix_prompt=prefix,
                    candidates=candidates,
                    state_kv_cache=state_kv_cache,
                    state_seq_len=state_seq_len,
                    temperature=settings.default_temperature
                )
                best_choice = max(prob_dict, key=prob_dict.get)
                confidence = calculate_choice_confidence(prob_dict)
                answers[qid] = ChoiceAnswer(
                    choice=best_choice,
                    probabilities=prob_dict,
                    confidence=round(confidence, 4)
                )

            elif isinstance(q, ScoreQuestion):
                prefix = f"Evaluation: {q.instructions}\nRubric Evaluation:"
                candidates = {k: f"Score {k} - {v}" for k, v in q.levels.items()}
                prob_dict = self._score_candidates(
                    prefix_prompt=prefix,
                    candidates=candidates,
                    state_kv_cache=state_kv_cache,
                    state_seq_len=state_seq_len,
                    temperature=settings.default_temperature
                )
                confidence = calculate_score_confidence(prob_dict)
                try:
                    expected_score = sum(float(k) * p for k, p in prob_dict.items())
                except ValueError:
                    expected_score = 1.0

                answers[qid] = ScoreAnswer(
                    score=round(expected_score, 2),
                    probabilities=prob_dict,
                    confidence=round(confidence, 4)
                )
            else:
                raise ValueError(f"Unsupported question type: {type(q)}")

        return SystemOneResponse(model=self.model_version, answers=answers)
    