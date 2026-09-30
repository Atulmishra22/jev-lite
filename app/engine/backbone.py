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

    @torch.no_grad()
    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        # Step 1: Ingest State ONCE to get KV-cache
        state_text = f"State Context:\n{self._format_state(request.state)}"
        formatted_state = self._format_chat(role="system", content=state_text, add_generation_prompt=False)
        state_inputs = self.tokenizer(formatted_state, return_tensors="pt").to(self.device)
        state_seq_len = state_inputs.input_ids.shape[1]

        state_outputs = self.backbone(**state_inputs, use_cache=True)
        state_kv_cache = state_outputs.past_key_values

        q_items = list(request.questions.items())
        prompts = []
        for qid, q in q_items:
            if isinstance(q, NoulQuestion):
                q_content = f"Statement: {q.statement}\nIs this statement true? (Answer Yes or No):"
            elif isinstance(q, ChoiceQuestion):
                options_str = "\n".join([f"- {k}: {v}" for k, v in q.criteria.items()])
                q_content = f"Question: {q.instructions}\nOptions:\n{options_str}\nSelected Option:"
            elif isinstance(q, ScoreQuestion):
                levels_str = "\n".join([f"- {k}: {v}" for k, v in q.levels.items()])
                q_content = f"Evaluation: {q.instructions}\nRubric:\n{levels_str}\nAnswer with the score number:"
            else:
                raise ValueError(f"Unsupported question type: {type(q)}")

            prompts.append(self._format_chat(role="user", content=q_content, add_generation_prompt=True))

        answers: dict[str, Answer] = {}
        chunk_size = settings.batch_chunk_size

        # stream in parallel chunks to avoid GPU VRAM OOM
        for i in range(0, len(q_items), chunk_size):
            chunk_q_items = q_items[i:i + chunk_size]
            chunk_prompts = prompts[i:i + chunk_size]
            current_batch_size = len(chunk_q_items)

            # Process this chunk in parallel
            q_inputs = self.tokenizer(chunk_prompts, padding=True, return_tensors="pt").to(self.device)

            # build full attention mask covering state + questions
            state_mask = torch.ones((current_batch_size, state_seq_len), dtype=torch.long, device=self.device)
            full_mask = torch.cat([state_mask, q_inputs.attention_mask], dim=1)

            # expand state KV cache to current batch size
            batched_kv = self._expand_kv_cache(state_kv_cache, current_batch_size)

            '''
            parallel foward pass for entir chunk
            '''
            q_out = self.backbone(input_ids=q_inputs.input_ids, attention_mask=full_mask, past_key_values=batched_kv)

            # extract answers for all questions in this chunk
            for idx, (qid, q) in enumerate(chunk_q_items):
                last_token_logits = q_out.logits[idx, -1, :]

                if isinstance(q, NoulQuestion):
                    yes_id = self._get_first_token_id("Yes")
                    no_id = self._get_first_token_id("No")
                    tf_logits = torch.tensor([last_token_logits[no_id].item(), last_token_logits[yes_id].item()])
                    prob = torch.softmax(tf_logits, dim=-1)[1].item()
                    answers[qid] = NoulAnswer(noul=round(prob, 4))

                elif isinstance(q, ChoiceQuestion):
                    option_names = list(q.criteria.keys())
                    cand_ids = [self._get_first_token_id(name) for name in option_names]
                    cand_logits = torch.tensor([last_token_logits[cid].item() for cid in cand_ids])
                    probs = torch.softmax(cand_logits, dim=-1).tolist()
                    prob_dict = {name: round(p, 4) for name, p in zip(option_names, probs)}
                    best_choice = max(prob_dict, key=prob_dict.get)
                    confidence = calculate_choice_confidence(prob_dict)
                    answers[qid] = ChoiceAnswer(
                        choice=best_choice,
                        probabilities=prob_dict,
                        confidence=round(confidence, 4)
                    )

                elif isinstance(q, ScoreQuestion):
                    level_names = list(q.levels.keys())
                    level_words = [q.levels[k].split()[0] for k in level_names]
                    level_ids = [self._get_first_token_id(w) for w in level_words]
                    level_logits = torch.tensor([last_token_logits[lid].item() for lid in level_ids])
                    probs = torch.softmax(level_logits, dim=-1).tolist()
                    prob_dict = {name: round(p, 4) for name, p in zip(level_names, probs)}
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

        return SystemOneResponse(model=self.model_version, answers=answers)
    