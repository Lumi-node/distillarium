from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from distillarium.engine.core import AttentionOnlyTransformer
from distillarium.engine.tokenizer import FunctionCallTokenizer
from distillarium.engine.schema import SchemaEncoder


JSON_CHARS = set('{}[]":,0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_.- \n\t')


class FunctionCallGenerator:
    def __init__(
        self,
        model: AttentionOnlyTransformer,
        tokenizer: FunctionCallTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def _build_allowed_mask(self, vocab_size: int) -> torch.Tensor:
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in range(vocab_size):
            try:
                decoded = self.tokenizer.decode([token_id])
                if all(c in JSON_CHARS for c in decoded):
                    mask[token_id] = True
            except Exception:
                pass
        mask[self.tokenizer.pad_token_id] = False
        return mask

    def generate(
        self,
        utterance: str,
        tools: list[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[dict]:
        self.model.eval()
        device = next(self.model.parameters()).device

        schema_encoder = SchemaEncoder(self.tokenizer)
        # SchemaEncoder.encode returns 1D; add batch dim for the model
        schema_ids = schema_encoder.encode(tools).unsqueeze(0).to(device)

        prompt = f"[QUERY] {utterance} [/QUERY] [CALL] "
        input_ids = self.tokenizer.encode(prompt).unsqueeze(0).to(device)

        generated: list[int] = []
        allowed_mask = self._build_allowed_mask(self.model.vocab_size).to(device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(input_ids, schema_ids)
                next_logits = logits[:, -1, :] / max(temperature, 1e-8)

                next_logits[:, ~allowed_mask] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()
                generated.append(token_id)

                decoded_so_far = self.tokenizer.decode(generated)
                if "[/CALL]" in decoded_so_far:
                    break

                input_ids = torch.cat([input_ids, next_token], dim=1)

        output_text = self.tokenizer.decode(generated)
        output_text = output_text.replace("[/CALL]", "").strip()

        return self._parse_output(output_text)

    def _parse_output(self, text: str) -> list[dict]:
        text = text.strip()
        for start in range(len(text)):
            if text[start] == "[":
                for end in range(len(text), start, -1):
                    try:
                        result = json.loads(text[start:end])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        continue
            if text[start] == "{":
                for end in range(len(text), start, -1):
                    try:
                        result = json.loads(text[start:end])
                        if isinstance(result, dict):
                            return [result]
                    except json.JSONDecodeError:
                        continue
        return []
