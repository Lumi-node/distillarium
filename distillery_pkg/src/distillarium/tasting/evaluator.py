"""Evaluator — computes Tasting Notes (held-out metrics + sample predictions)."""

from __future__ import annotations

import json
from typing import Any


def _strip_keys_values(obj: Any) -> Any:
    """Recursively strip whitespace from dict keys + string values.

    The WordPiece-based tokenizer inserts spaces between subword pieces,
    which leaks into JSON keys/values on decode. We normalize.
    """
    if isinstance(obj, dict):
        return {
            (k.strip() if isinstance(k, str) else k):
            _strip_keys_values(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_keys_values(v) for v in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def parse_generated_call(generated_calls: list[dict]) -> dict | None:
    """Pull a single normalized (name, args) prediction or None."""
    if not generated_calls:
        return None
    first = _strip_keys_values(generated_calls[0])
    if not isinstance(first, dict):
        return None
    name = first.get("name")
    args = first.get("args", {})
    if not isinstance(name, str):
        return None
    return {"name": name, "args": args if isinstance(args, dict) else {}}


def evaluate(
    generator,
    eval_data: list[tuple[str, list[dict], str]],
    max_examples: int = 100,
) -> dict:
    """Run inference on `eval_data` and compute Tasting Notes.

    Returns a metrics dict + sample predictions for the report.
    """
    n_total = 0
    n_tool_correct = 0
    n_exact_call = 0
    arg_key_tp = 0
    arg_key_fp = 0
    arg_key_fn = 0

    samples = []  # (utt, gold, pred, verdict)
    n_to_eval = min(max_examples, len(eval_data))

    for utt, tools, gold_json in eval_data[:n_to_eval]:
        n_total += 1
        try:
            gold = json.loads(gold_json)
            gold_call = gold[0] if gold else None
            if gold_call is None:
                continue

            predicted = generator.generate(utt, tools)
            pred = parse_generated_call(predicted)

            verdict = "wrong"
            if pred is None:
                arg_key_fn += len(gold_call.get("args", {}))
            else:
                if pred["name"] == gold_call["name"]:
                    n_tool_correct += 1

                gold_keys = set(gold_call.get("args", {}).keys())
                pred_keys = set(pred["args"].keys())
                arg_key_tp += len(gold_keys & pred_keys)
                arg_key_fp += len(pred_keys - gold_keys)
                arg_key_fn += len(gold_keys - pred_keys)

                if (
                    pred["name"] == gold_call["name"]
                    and gold_keys == pred_keys
                    and all(
                        str(pred["args"].get(k)) == str(gold_call["args"].get(k))
                        for k in gold_keys
                    )
                ):
                    n_exact_call += 1
                    verdict = "exact"
                elif pred["name"] == gold_call["name"]:
                    verdict = "tool_only"

            if len(samples) < 8:
                samples.append({
                    "utterance": utt,
                    "gold": gold_call,
                    "predicted": pred,
                    "verdict": verdict,
                })
        except Exception:
            continue

    p = arg_key_tp / max(1, arg_key_tp + arg_key_fp)
    r = arg_key_tp / max(1, arg_key_tp + arg_key_fn)
    f1 = 2 * p * r / max(1e-9, p + r)

    return {
        "n_evaluated": n_total,
        "tool_name_accuracy": round(n_tool_correct / max(1, n_total), 3),
        "exact_call_accuracy": round(n_exact_call / max(1, n_total), 3),
        "arg_key_precision": round(p, 3),
        "arg_key_recall": round(r, 3),
        "arg_key_f1": round(f1, 3),
        "samples": samples,
    }
