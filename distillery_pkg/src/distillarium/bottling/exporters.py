"""Exporters — bottle a Spirit into a deployable format.

v0.1 ships PyTorch native. ONNX is implemented but requires the `[onnx]` extra.
GGUF and browser-WASM are planned for v0.2.
"""

from __future__ import annotations

from pathlib import Path


def bottle_pytorch(spirit, out: str | Path) -> Path:
    """Save the Spirit as a single .pt file (PyTorch native)."""
    return spirit.save(out)


def bottle_onnx(spirit, out: str | Path) -> Path:
    """Export the Spirit's model as ONNX for cross-runtime inference.

    Requires: pip install 'distillarium[onnx]'
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as e:
        raise RuntimeError("torch is required for ONNX export") from e

    model = spirit.model
    model.eval()

    seq_len = spirit.recipe.student.max_seq_len
    schema_len = 96
    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_schema = torch.zeros(1, schema_len, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input, dummy_schema),
        str(out),
        input_names=["input_ids", "schema_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "schema_ids": {0: "batch", 1: "schema"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    return out
