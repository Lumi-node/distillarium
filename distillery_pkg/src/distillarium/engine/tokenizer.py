from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "[TOOLS]", "[/TOOLS]", "[TOOL]", "[/TOOL]",
    "[CALL]", "[/CALL]", "[QUERY]", "[/QUERY]",
]


class FunctionCallTokenizer:
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._trained = False

    @property
    def pad_token_id(self) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.pad_token_id or 0
        return 0

    def train(self, corpus: list[str]) -> None:
        base = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        base.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=1,
        )
        base.train_from_iterator(corpus, trainer=trainer)

        with tempfile.TemporaryDirectory() as tmpdir:
            tok_path = Path(tmpdir) / "tokenizer.json"
            base.save(str(tok_path))
            self._tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tok_path),
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
            )
        self._trained = True

    def encode(self, text: str) -> torch.Tensor:
        if not self._trained or self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        encoded = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].squeeze(0)

    def encode_to_ids(self, text: str) -> list[int]:
        return self.encode(text).tolist()

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if not self._trained or self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def get_vocab_size(self) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.vocab_size
        return self.vocab_size

    def get_hf_tokenizer(self) -> PreTrainedTokenizerFast:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        return self._tokenizer
