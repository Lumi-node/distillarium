"""Spirit — the bottled output of a distillation run.

A Spirit bundles model weights + tokenizer + recipe + metrics so it can be
deployed, audited, forked, and shared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from distillarium.engine.core import AttentionOnlyTransformer
from distillarium.engine.tokenizer import FunctionCallTokenizer
from distillarium.recipe import Recipe


@dataclass
class Spirit:
    """A bottled, deployable Spirit."""
    name: str
    recipe: Recipe
    model: AttentionOnlyTransformer
    tokenizer: FunctionCallTokenizer
    metrics: dict[str, Any] = field(default_factory=dict)
    loss_curve: list[float] = field(default_factory=list)
    n_params: int = 0

    @property
    def proof(self) -> int:
        """The headline proof — main task metric × 100, rounded."""
        for key in ("tool_name_accuracy", "exact_call_accuracy", "macro_f1", "accuracy"):
            if key in self.metrics:
                v = self.metrics[key]
                if isinstance(v, (int, float)):
                    return round(v * 100 if v <= 1.0 else v)
        return 0

    def save(self, path: str | Path) -> Path:
        """Save the Spirit as a single .pt file (PyTorch native)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "recipe": self.recipe.to_dict(),
                "model_state_dict": self.model.state_dict(),
                "model_config": {
                    "vocab_size": self.model.vocab_size,
                    "d_model": self.recipe.student.d_model,
                    "n_heads": self.recipe.student.n_heads,
                    "n_layers": self.recipe.student.n_layers,
                    "max_seq_len": self.recipe.student.max_seq_len,
                },
                "tokenizer_state": self._serialize_tokenizer(),
                "metrics": self.metrics,
                "loss_curve": self.loss_curve,
                "n_params": self.n_params,
                "distillarium_version": "0.1.0",
            },
            path,
        )
        return path

    def _serialize_tokenizer(self) -> dict:
        """Serialize the tokenizer's HF backing object to bytes."""
        if self.tokenizer._tokenizer is None:
            return {}
        # PreTrainedTokenizerFast can save to a temp dir then we read the json
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tokenizer._tokenizer.save_pretrained(tmpdir)
            files: dict[str, bytes] = {}
            for p in Path(tmpdir).iterdir():
                files[p.name] = p.read_bytes()
            return {"hf_files": files, "vocab_size": self.tokenizer.vocab_size}


def load_spirit(path: str | Path) -> Spirit:
    """Load a saved Spirit from a .pt file."""
    path = Path(path)
    ckpt = torch.load(path, weights_only=False)
    recipe = Recipe.from_dict(ckpt["recipe"])
    mc = ckpt["model_config"]
    model = AttentionOnlyTransformer(
        vocab_size=mc["vocab_size"],
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        max_seq_len=mc["max_seq_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = _deserialize_tokenizer(ckpt.get("tokenizer_state", {}))

    return Spirit(
        name=ckpt.get("name", path.stem),
        recipe=recipe,
        model=model,
        tokenizer=tokenizer,
        metrics=ckpt.get("metrics", {}),
        loss_curve=ckpt.get("loss_curve", []),
        n_params=ckpt.get("n_params", 0),
    )


def _deserialize_tokenizer(state: dict) -> FunctionCallTokenizer:
    """Rebuild the FunctionCallTokenizer from a serialized state."""
    import tempfile
    from transformers import PreTrainedTokenizerFast

    if not state or "hf_files" not in state:
        return FunctionCallTokenizer(vocab_size=state.get("vocab_size", 4096))

    tok = FunctionCallTokenizer(vocab_size=state.get("vocab_size", 4096))
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, blob in state["hf_files"].items():
            (Path(tmpdir) / name).write_bytes(blob)
        tok._tokenizer = PreTrainedTokenizerFast.from_pretrained(tmpdir)
        tok._trained = True
    return tok
