from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

import torch


class SchemaEncoder:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def _serialize_tools(self, tool_definitions: list[dict]) -> str:
        parts = ["[TOOLS]"]
        for tool in tool_definitions:
            name = tool.get("name", "unknown")
            params = tool.get("params", tool.get("parameters", {}))
            desc = tool.get("description", "")
            parts.append(f"[TOOL] {name}")
            if desc:
                parts.append(f"  description: {desc}")
            if isinstance(params, dict):
                for pname, pinfo in params.items():
                    if isinstance(pinfo, dict):
                        ptype = pinfo.get("type", "string")
                        pdesc = pinfo.get("description", "")
                        preq = pinfo.get("required", False)
                        parts.append(
                            f"  param: {pname} type={ptype} required={preq} {pdesc}"
                        )
                    else:
                        parts.append(f"  param: {pname} value={pinfo}")
            parts.append("[/TOOL]")
        parts.append("[/TOOLS]")
        return "\n".join(parts)

    def encode(self, tool_definitions: list[dict]) -> torch.Tensor:
        """Encode tool schemas to a 1D LongTensor of token ids.

        Always returns 1D (shape [L]). Callers that need batch dimension should
        stack or unsqueeze explicitly. This is the canonical schema encoding
        path — trainer.py and inference.py both go through here.
        """
        text = self._serialize_tools(tool_definitions)
        # FunctionCallTokenizer.encode returns 1D tensor (the project's own tokenizer)
        if hasattr(self.tokenizer, "encode") and not hasattr(self.tokenizer, "encode_to_ids"):
            # Plain HF tokenizer path: call as __call__ and squeeze the batch dim
            encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            return encoded["input_ids"].squeeze(0)
        # FunctionCallTokenizer path (has both encode and encode_to_ids)
        ids = self.tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            return ids if ids.dim() == 1 else ids.squeeze(0)
        return torch.tensor(ids, dtype=torch.long)
