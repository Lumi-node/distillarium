"""BFCL alignment — load Berkeley Function Calling Leaderboard examples and
convert them into the (utterance, tools, gold_json) tuples the evaluator uses.

BFCL ships several splits as JSONL on the gorilla repo:

  https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/data/

This module's job is the format translation, NOT downloading. Pass it a path
to a `BFCL_v3_simple.json` (or similar single-turn AST-eval split) and it
yields the same 3-tuples the rest of our evaluator consumes.

We deliberately scope to the **single-turn AST-eval categories** for v0.2:

  - BFCL_v3_simple                — 1 utterance, 1 tool to choose from N, 1 call
  - BFCL_v3_multiple              — 1 utterance, multiple tools available
  - BFCL_v3_parallel              — 1 utterance, multiple calls in one go

The chat-multi-turn / executable / Java / JavaScript splits are out of scope
for v0.2 (the Needle student doesn't speak those execution sandboxes).

What we report when scored against BFCL:

  tool_name_accuracy   ↔  BFCL "function" pass rate (AST level)
  exact_call_accuracy  ↔  BFCL "AST" pass rate (function + args + arg values)
  arg_key_f1            — our own, not a BFCL-native metric

The plan for v0.2 is to publish those three numbers on Needle alongside the
internal eval cuts, so the comparison to TinyAgent / DistillKit is honest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


# BFCL stores AST gold as a list of dicts where the SINGLE key is the function
# name and the value is the args dict. Their `parallel` split has multiple.
# Our evaluator expects `[{"name": ..., "args": ...}, ...]`.

def _bfcl_call_to_ours(bfcl_call: dict) -> dict:
    """{"send_message": {"contact": "mom"}}  ->  {"name": "send_message", "args": {"contact": "mom"}}"""
    if not isinstance(bfcl_call, dict) or len(bfcl_call) != 1:
        raise ValueError(f"Unexpected BFCL call shape: {bfcl_call!r}")
    [(name, args)] = bfcl_call.items()
    if not isinstance(args, dict):
        raise ValueError(f"BFCL args for {name!r} is not a dict: {args!r}")
    return {"name": name, "args": args}


def _bfcl_function_to_tool(fn: dict) -> dict:
    """BFCL `function` entry → our tool schema (name + params).

    BFCL function entries look like:
      {"name": "...", "description": "...",
       "parameters": {"type": "dict",
                      "properties": {"key": {"type": "...", ...}, ...},
                      "required": [...]}}
    """
    params = fn.get("parameters", {})
    properties = params.get("properties", {}) if isinstance(params, dict) else {}
    required = set(params.get("required", []) or [])
    out_params = {}
    for key, spec in properties.items():
        out_params[key] = {
            "type": (spec or {}).get("type", "string"),
            "required": key in required,
            "description": (spec or {}).get("description", ""),
        }
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "params": out_params,
    }


def load_bfcl_split(
    path: str | Path,
    split: str | None = None,
) -> Iterator[tuple[str, list[dict], str]]:
    """Yield (utterance, tools, gold_json) tuples from a BFCL JSONL file.

    Args:
        path: file path to a BFCL split file (one example per line).
        split: optional label; only used for error messages.

    The file is expected to contain one JSON object per line with these keys:
        question: list[list[{"role": "user", "content": str}]]
        function: list[BFCL function schema dicts]
        ground_truth: list[BFCL call dicts]  (one per call, multi for parallel)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"BFCL split not found at {p}. Download with:\n"
            f"  curl -sL https://raw.githubusercontent.com/ShishirPatil/gorilla/"
            f"main/berkeley-function-call-leaderboard/data/BFCL_v3_simple.json "
            f"-o {p}"
        )

    label = split or p.stem
    line_no = 0
    for raw in p.read_text().splitlines():
        line_no += 1
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{label}:{line_no}: invalid JSON: {e}") from e

        # `question` is a list of conversations; for single-turn we take the
        # first user message of the first conversation
        question = row.get("question") or []
        if not question or not isinstance(question[0], list) or not question[0]:
            continue
        first_msg = question[0][0]
        utterance = (first_msg or {}).get("content")
        if not isinstance(utterance, str) or not utterance.strip():
            continue

        # `function` is the available tool set for this example
        fns = row.get("function") or []
        tools = [_bfcl_function_to_tool(fn) for fn in fns if isinstance(fn, dict)]

        # `ground_truth` is the gold AST calls
        gold_raw = row.get("ground_truth") or []
        gold = []
        try:
            for c in gold_raw:
                gold.append(_bfcl_call_to_ours(c))
        except ValueError:
            # Skip rows we can't reliably parse rather than poison the eval
            continue
        if not gold:
            continue

        yield (utterance, tools, json.dumps(gold))


def score_against_bfcl(generator, path: str | Path, max_examples: int = 100) -> dict:
    """Convenience wrapper: load a BFCL split + score `generator` on it.

    Returns the same Tasting Notes shape as `evaluate()`, plus a `bfcl_split`
    field naming the split file so reports stay reproducible.
    """
    from distillarium.tasting.evaluator import evaluate

    eval_data = list(load_bfcl_split(path))
    metrics = evaluate(generator, eval_data, max_examples=max_examples)
    metrics["bfcl_split"] = str(Path(path).name)
    return metrics
