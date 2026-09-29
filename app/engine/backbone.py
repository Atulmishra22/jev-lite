import json
from typing import Any, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from app.core.config import settings
from app.engine.heads import NoulHead, ScoreHead, ChoiceHead
from app.engine.calibration import (calculate_choice_confidence, calculate_score_confidence)
from app.schemas.request import SystemOneRequest, ChoiceQuestion, ScoreQuestion, NoulQuestion
from app.schemas.response import SystemOneResponse, ChoiceAnswer, ScoreAnswer, NoulAnswer, Answer

class SystemOneEngine(nn.Module):
    def __init__(self, model_name : str = settings.model_name, device: str = settings.device):

        super().__init__()
        self.device = torch.device(device)
        self.model_version = settings.model_version

        # 1. Load Tokenizer & Base Transformer Backbone (No LM text-generation head!)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.to(self.device)
        self.backbone.eval()

        hidden_size = self.backbone.config.hidden_size

        # 2. Attach our system one  decision heads
        dtype = self.backbone.dtype
        self.noul_head = NoulHead(hidden_size).to(self.device, dtype=dtype)
        self.score_head = ScoreHead(hidden_size).to(self.device, dtype=dtype)
        self.choice_head = ChoiceHead(hidden_size).to(self.device, dtype=dtype)

    def _format_state(self, state: Union[str, dict[str,Any], list[Any]]) -> str:
        if isinstance(state, str):
            return state
        return json.dumps(state, indent=2)

    def _format_chat(self, role: str, content:str, add_generation_prompt: bool = False) -> str:
        """ Uses tokenizer's official chat template for aximum portability."""
        messages = [{"role": role, "content": content}]
        return self.tokenizer.apply_chat_template(messages, tokenize= False, add_generation_prompt=add_generation_prompt)

    def _get_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.backbone(**inputs)
        return outputs.last_hidden_state[:, -1, :].squeeze(0)

    @torch.no_grad()
    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        # step 1: format state using the model's official chat template
        state_text = F"State Context:\n{ self._format_state(request.state)}"
        formatted_state = self._format_chat(role="system", content=state_text,add_generation_prompt=False)
        state_inputs = self.tokenizer(formatted_state, return_tensors="pt").to(self.device)

        # Ingest state Once to get KV-cache
        state_outputs = self.backbone(**state_inputs,use_cache=True)
        state_kv_cache = state_outputs.past_key_values

        answers: dict[str, Answer] = {}

        # step 2 : evaluate each question against the cached state
        for q_id, question in request.questions.items():
            if isinstance(question, NoulQuestion):
                q_content = F"Statement: {question.statement}\n Is this statemnet true?"
                q_text = self._format_chat(role="user", content=q_content, add_generation_prompt= True)
                q_inputs = self.tokenizer(q_text, return_tensors="pt").to(self.device)

                q_out = self.backbone(**q_inputs, past_key_values= state_kv_cache)
                hidden = q_out.last_hidden_state[:, -1, :]
                prob = self.noul_head(hidden).item()

                answers[q_id] = NoulAnswer(noul=round(prob, 4))

            elif isinstance(question, ChoiceQuestion):
                q_content = f"Question: {question.instructions}\nSelect the best option from the criteria."
                q_text = self._format_chat(role="user", content=q_content, add_generation_prompt=True)
                q_inputs = self.tokenizer(q_text, return_tensors="pt").to(self.device)
                q_out = self.backbone(**q_inputs, past_key_values=state_kv_cache)
                query_hidden = q_out.last_hidden_state[:, -1, :].squeeze(0)
                option_names = list(question.criteria.keys())
                option_texts = [f"{k}: {v}" for k, v in question.criteria.items()]
                option_hiddens = torch.stack([self._get_embedding(t) for t in option_texts])
                probs_tensor = self.choice_head(query_hidden, option_hiddens, temperature=settings.default_temperature)
                probs_list = probs_tensor.tolist()
                prob_dict = {name: round(p, 4) for name, p in zip(option_names, probs_list)}
                best_choice = max(prob_dict, key=prob_dict.get)
                confidence = calculate_choice_confidence(prob_dict)
                answers[q_id] = ChoiceAnswer(
                    choice=best_choice,
                    probabilities=prob_dict,
                    confidence=round(confidence, 4)
                )
            elif isinstance(question, ScoreQuestion):
                q_content = f"Evaluate score: {question.instructions}"
                q_text = self._format_chat(role="user", content=q_content, add_generation_prompt=True)
                q_inputs = self.tokenizer(q_text, return_tensors="pt").to(self.device)
                q_out = self.backbone(**q_inputs, past_key_values=state_kv_cache)
                query_hidden = q_out.last_hidden_state[:, -1, :]
                raw_score = self.score_head(query_hidden).item()
                level_names = list(question.levels.keys())
                level_texts = [f"Level {k}: {v}" for k, v in question.levels.items()]
                level_hiddens = torch.stack([self._get_embedding(t) for t in level_texts])
                level_probs = self.choice_head(query_hidden.squeeze(0), level_hiddens).tolist()
                prob_dict = {name: round(p, 4) for name, p in zip(level_names, level_probs)}
                confidence = calculate_score_confidence(prob_dict)
                answers[q_id] = ScoreAnswer(
                    score=round(raw_score, 2),
                    probabilities=prob_dict,
                    confidence=round(confidence, 4)
                )
        return SystemOneResponse(model=self.model_version, answers=answers)   
