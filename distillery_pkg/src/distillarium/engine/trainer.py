from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from distillarium.engine.core import AttentionOnlyTransformer
from distillarium.engine.schema import SchemaEncoder
from distillarium.engine.tokenizer import FunctionCallTokenizer

logger = logging.getLogger(__name__)


class FunctionCallTrainer:
    def __init__(
        self,
        model: AttentionOnlyTransformer,
        tokenizer: FunctionCallTokenizer,
        lr: float = 3e-4,
        schema_encoder: SchemaEncoder | None = None,
        schema_max_len: int = 96,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.loss_history: list[float] = []
        # SchemaEncoder is the canonical path for encoding available tools.
        # If not provided, build one over the same tokenizer.
        self.schema_encoder = schema_encoder or SchemaEncoder(tokenizer)
        self.schema_max_len = schema_max_len

    def _pad_or_truncate(self, ids: torch.Tensor, length: int, pad_id: int) -> torch.Tensor:
        if ids.dim() != 1:
            ids = ids.view(-1)
        if len(ids) >= length:
            return ids[:length]
        pad = torch.full((length - len(ids),), pad_id, dtype=torch.long)
        return torch.cat([ids, pad])

    def _encode_schema_from_tools(self, tools: list[dict] | None) -> torch.Tensor:
        """Encode the AVAILABLE tools (not the target). No label leakage."""
        pad_id = self.tokenizer.pad_token_id
        if tools is None or len(tools) == 0:
            # Fallback: a sentinel "unknown tools" schema. Should rarely fire
            # in real training — the data pipeline must supply tool defs.
            s_ids = self.tokenizer.encode("[TOOLS] [/TOOLS]")
        else:
            s_ids = self.schema_encoder.encode(tools)
        return self._pad_or_truncate(s_ids, self.schema_max_len, pad_id)

    def _prepare_batch(
        self,
        utterances: list[str],
        targets: list[str],
        max_len: int = 256,
        tools_per_example: list[list[dict]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids_list = []
        target_ids_list = []
        schema_ids_list = []
        pad_id = self.tokenizer.pad_token_id

        for i, (utt, tgt) in enumerate(zip(utterances, targets)):
            full_text = f"[QUERY] {utt} [/QUERY] [CALL] {tgt} [/CALL]"
            ids = self.tokenizer.encode(full_text)
            if len(ids) > max_len:
                ids = ids[:max_len]

            inp = ids[:-1]
            tgt_ids = ids[1:]

            pad_len = max_len - 1 - len(inp)

            if pad_len > 0:
                inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
                tgt_ids = torch.cat([tgt_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            else:
                inp = inp[: max_len - 1]
                tgt_ids = tgt_ids[: max_len - 1]

            tools = tools_per_example[i] if tools_per_example is not None else None
            s_ids = self._encode_schema_from_tools(tools)

            input_ids_list.append(inp)
            target_ids_list.append(tgt_ids)
            schema_ids_list.append(s_ids)

        return (
            torch.stack(input_ids_list),
            torch.stack(target_ids_list),
            torch.stack(schema_ids_list),
        )

    def train(
        self,
        train_data: list[tuple[str, str]] | list[tuple[str, list[dict], str]],
        epochs: int = 3,
        batch_size: int = 8,
        max_len: int = 256,
    ) -> list[float]:
        self.model.train()
        device = next(self.model.parameters()).device

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            indices = list(range(len(train_data)))
            import random
            random.shuffle(indices)

            for i in tqdm(
                range(0, len(indices), batch_size),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
            ):
                batch_idx = indices[i : i + batch_size]
                # Support both (utt, target) and (utt, tools, target) row shapes
                has_tools = len(train_data[batch_idx[0]]) == 3
                if has_tools:
                    utterances = [train_data[j][0] for j in batch_idx]
                    tools_pe = [train_data[j][1] for j in batch_idx]
                    targets = [train_data[j][2] for j in batch_idx]
                else:
                    utterances = [train_data[j][0] for j in batch_idx]
                    targets = [train_data[j][1] for j in batch_idx]
                    tools_pe = None

                input_ids, target_ids, schema_ids = self._prepare_batch(
                    utterances, targets, max_len, tools_per_example=tools_pe
                )
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                schema_ids = schema_ids.to(device)

                logits = self.model(input_ids, schema_ids)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)
            logger.info(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

        return self.loss_history

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.loss_history = ckpt.get("loss_history", [])
