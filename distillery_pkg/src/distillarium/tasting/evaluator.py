"""Evaluator — Tasting Notes, with optional teacher-baseline & version regression.

The `evaluate()` function answers three honest questions in one pass:

  1. How well does the student do on held-out cuts?      (always)
  2. How well does the teacher do on the SAME cuts?      (if `teacher` is given)
     - i.e. is the gap student-vs-teacher, or is the gold labeling itself noisy?
  3. Did anything regress vs. the previous run?           (if `previous` is given)
     - i.e. did v2 of this recipe accidentally make things worse?

All three pieces are optional and additive. v0.1 callers keep working unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Metrics where a HIGHER value is better. Anything not in this set is treated
# as "lower is better" (e.g. final_loss).
_HIGHER_IS_BETTER = {
    "tool_name_accuracy",
    "exact_call_accuracy",
    "arg_key_precision",
    "arg_key_recall",
    "arg_key_f1",
}

# Tolerance below which a metric drop is reported as flat, not a regression.
# Held-out evals on 100 samples have ~3pp natural variance; 0.01 is generous.
_REGRESSION_TOLERANCE = 0.01


class TeacherEvalGenerator:
    """Adapter: makes a Teacher quack like a student generator for `evaluate()`.

    The student generator interface is `.generate(utt, tools) -> list[dict]`.
    Teachers natively expose data-generation (`generate_batch`), not inference.
    This adapter forwards `.generate(...)` to the teacher's `answer(...)` so the
    same evaluator code path can score either side.
    """

    def __init__(self, teacher, label: str = "teacher"):
        if not hasattr(teacher, "answer"):
            raise TypeError(
                f"{type(teacher).__name__} cannot be used as an eval generator: "
                "no answer() method. Override Teacher.answer() to enable this."
            )
        self._teacher = teacher
        self.label = label

    def generate(self, utterance: str, tools: list[dict]) -> list[dict]:
        return self._teacher.answer(utterance, tools)


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


def _score_one_run(generator, eval_data, max_examples: int) -> dict:
    """Run a single generator over `eval_data` and return raw metrics + samples.

    Pulled out of `evaluate()` so both student and teacher use the same scoring.
    """
    n_total = 0
    n_tool_correct = 0
    n_exact_call = 0
    arg_key_tp = 0
    arg_key_fp = 0
    arg_key_fn = 0

    samples = []
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


def _load_previous(previous: dict | str | Path) -> dict:
    """Accept either a metrics dict or a path to a JSON file containing one."""
    if isinstance(previous, dict):
        return previous
    path = Path(previous)
    payload = json.loads(path.read_text())
    # Allow either a bare metrics dict OR a Tasting Notes-shaped dict with
    # a top-level "student" / "metrics" key.
    if "student" in payload and isinstance(payload["student"], dict):
        return payload["student"]
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return payload["metrics"]
    return payload


def _compute_deltas(current: dict, baseline: dict) -> dict:
    """Per-metric delta (current - baseline) for numeric fields they share.

    Positive delta = current is higher than baseline. For "higher is better"
    metrics that means current improved; for "lower is better" it means
    current got worse. The `regressions` list is the human-readable summary.
    """
    deltas: dict[str, float] = {}
    regressions: list[str] = []

    for key, cur_val in current.items():
        if not isinstance(cur_val, (int, float)):
            continue
        if key not in baseline:
            continue
        base_val = baseline[key]
        if not isinstance(base_val, (int, float)):
            continue

        delta = round(cur_val - base_val, 4)
        deltas[key] = delta

        higher_better = key in _HIGHER_IS_BETTER
        # A regression is: dropped on a higher-is-better metric, OR rose on a
        # lower-is-better metric, by more than _REGRESSION_TOLERANCE.
        if higher_better and delta < -_REGRESSION_TOLERANCE:
            regressions.append(
                f"{key}: {base_val:.3f} → {cur_val:.3f} ({delta:+.3f})"
            )
        elif (not higher_better) and delta > _REGRESSION_TOLERANCE:
            regressions.append(
                f"{key}: {base_val:.3f} → {cur_val:.3f} ({delta:+.3f})"
            )

    return {"deltas": deltas, "regressions": regressions}


def evaluate(
    generator,
    eval_data: list[tuple[str, list[dict], str]],
    max_examples: int = 100,
    teacher=None,
    teacher_max_examples: int | None = None,
    previous: dict | str | Path | None = None,
) -> dict:
    """Run inference on `eval_data` and compute Tasting Notes.

    Args:
        generator: object with `.generate(utt, tools) -> list[dict]` (the student).
        eval_data: list of (utterance, tools, gold_json) triples.
        max_examples: cap how many cuts to score.
        teacher: optional Teacher instance to compute a teacher-baseline. Wrapped
            with `TeacherEvalGenerator` and scored against the SAME cuts. Adds
            `teacher_metrics` and `delta_vs_teacher` to the returned dict.
        teacher_max_examples: optional smaller cap for the teacher pass (each
            teacher call costs API tokens; default = same cap as student).
        previous: optional previous metrics dict, OR path to a JSON file with
            one (a bare metrics dict, or a Tasting Notes payload with a
            top-level "student" / "metrics" key). Adds `regression` to the
            returned dict.

    Returns:
        Tasting Notes dict. Always includes the student-side metrics
        (back-compat with v0.1). When `teacher` is given, also includes:
            teacher_metrics: dict of teacher's metrics on the same cuts
            delta_vs_teacher: per-metric (student - teacher); negative on a
                higher-is-better metric = student is below teacher
        When `previous` is given, also includes:
            regression: {previous, deltas, regressions[]}
    """
    student_metrics = _score_one_run(generator, eval_data, max_examples)

    out: dict[str, Any] = dict(student_metrics)

    if teacher is not None:
        teacher_gen = (
            teacher if hasattr(teacher, "generate") and not hasattr(teacher, "answer")
            else TeacherEvalGenerator(teacher)
        )
        cap = teacher_max_examples if teacher_max_examples is not None else max_examples
        teacher_metrics = _score_one_run(teacher_gen, eval_data, cap)
        out["teacher_metrics"] = teacher_metrics
        out["delta_vs_teacher"] = _compute_deltas(student_metrics, teacher_metrics)["deltas"]

    if previous is not None:
        baseline = _load_previous(previous)
        diff = _compute_deltas(student_metrics, baseline)
        out["regression"] = {
            "previous": baseline,
            "deltas": diff["deltas"],
            "regressions": diff["regressions"],
        }

    return out
