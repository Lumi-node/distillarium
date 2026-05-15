"""Tests for evaluator additions in v0.2-prep:

  - student-only baseline still works (back-compat with 0.1)
  - teacher-baseline pass via TeacherEvalGenerator
  - version regression detection via `previous=`

We don't hit the real Gemini API here — we use a deterministic in-memory fake
generator and a deterministic in-memory fake teacher.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------- Fake generators / teachers (no external services) ----------

class _DictDrivenGenerator:
    """A generator that returns a pre-canned answer per utterance.

    Quacks like FunctionCallGenerator: `.generate(utt, tools) -> list[dict]`.
    """

    def __init__(self, table: dict[str, list[dict]]):
        self.table = table

    def generate(self, utt: str, tools: list[dict]) -> list[dict]:
        return self.table.get(utt, [])


class _PerfectTeacher:
    """A fake teacher that returns the gold call verbatim — used to verify
    that `evaluate()` scores a 'teacher' separately from the student."""

    def __init__(self, gold_table: dict[str, list[dict]]):
        self.gold_table = gold_table

    def answer(self, utt: str, tools: list[dict]) -> list[dict]:
        return self.gold_table.get(utt, [])


# ---------- Shared eval fixture ----------

def _eval_data():
    """Three (utt, tools, gold_json) triples — enough to test all metric paths."""
    tools = [
        {"name": "send_message", "params": {"contact": {"type": "string"}}},
        {"name": "set_timer", "params": {"minutes": {"type": "integer"}}},
    ]
    return [
        ("text mom hi",                      tools, json.dumps([{"name": "send_message", "args": {"contact": "mom"}}])),
        ("start a 5 minute timer",           tools, json.dumps([{"name": "set_timer",    "args": {"minutes": 5}}])),
        ("text dad about dinner",            tools, json.dumps([{"name": "send_message", "args": {"contact": "dad"}}])),
    ]


# ---------- Student-only path (back-compat) ----------

def test_evaluate_student_only_back_compat():
    from distillarium.tasting import evaluate

    data = _eval_data()
    perfect_student = _DictDrivenGenerator({
        "text mom hi":            [{"name": "send_message", "args": {"contact": "mom"}}],
        "start a 5 minute timer": [{"name": "set_timer",    "args": {"minutes": 5}}],
        "text dad about dinner":  [{"name": "send_message", "args": {"contact": "dad"}}],
    })

    metrics = evaluate(perfect_student, data, max_examples=10)

    assert metrics["n_evaluated"] == 3
    assert metrics["tool_name_accuracy"] == 1.0
    assert metrics["exact_call_accuracy"] == 1.0
    # Back-compat: no teacher or regression keys when not asked for
    assert "teacher_metrics" not in metrics
    assert "regression" not in metrics


# ---------- Teacher-baseline path ----------

def test_evaluate_with_teacher_baseline():
    """A weak student + a perfect teacher → delta_vs_teacher shows the gap."""
    from distillarium.tasting import evaluate

    data = _eval_data()
    # Student gets the names right but always drops the args (mimics 0.1.1 Needle)
    weak_student = _DictDrivenGenerator({
        "text mom hi":            [{"name": "send_message", "args": {}}],
        "start a 5 minute timer": [{"name": "set_timer",    "args": {}}],
        "text dad about dinner":  [{"name": "send_message", "args": {}}],
    })
    perfect_teacher = _PerfectTeacher({
        "text mom hi":            [{"name": "send_message", "args": {"contact": "mom"}}],
        "start a 5 minute timer": [{"name": "set_timer",    "args": {"minutes": 5}}],
        "text dad about dinner":  [{"name": "send_message", "args": {"contact": "dad"}}],
    })

    metrics = evaluate(weak_student, data, max_examples=10, teacher=perfect_teacher)

    # Student got tool names right but blew exact-call
    assert metrics["tool_name_accuracy"] == 1.0
    assert metrics["exact_call_accuracy"] == 0.0

    # Teacher metrics are recorded
    tm = metrics["teacher_metrics"]
    assert tm["tool_name_accuracy"] == 1.0
    assert tm["exact_call_accuracy"] == 1.0

    # delta_vs_teacher reports the gap (negative on higher-better metrics)
    delta = metrics["delta_vs_teacher"]
    assert delta["exact_call_accuracy"] == -1.0
    assert delta["tool_name_accuracy"] == 0.0


def test_teacher_eval_generator_rejects_objects_without_answer():
    from distillarium.tasting import TeacherEvalGenerator

    class NoAnswer:
        pass

    with pytest.raises(TypeError, match="no answer"):
        TeacherEvalGenerator(NoAnswer())


def test_teacher_eval_generator_forwards_to_answer():
    from distillarium.tasting import TeacherEvalGenerator

    class _T:
        def answer(self, utt, tools):
            return [{"name": "echo", "args": {"utt": utt, "n_tools": len(tools)}}]

    gen = TeacherEvalGenerator(_T(), label="echo-teacher")
    out = gen.generate("hi", [{"name": "a"}, {"name": "b"}])
    assert out == [{"name": "echo", "args": {"utt": "hi", "n_tools": 2}}]


# ---------- Version regression path ----------

def test_evaluate_with_previous_metrics_detects_regression():
    from distillarium.tasting import evaluate

    data = _eval_data()
    # Student that gets only 1 of 3 right (33%)
    sloppy_student = _DictDrivenGenerator({
        "text mom hi":            [{"name": "send_message", "args": {"contact": "mom"}}],
        "start a 5 minute timer": [{"name": "send_message", "args": {}}],  # wrong tool
        "text dad about dinner":  [],                                       # no answer
    })

    # Previous run: student scored 95% across the board
    previous = {
        "tool_name_accuracy": 0.95,
        "exact_call_accuracy": 0.95,
        "arg_key_f1": 0.95,
    }

    metrics = evaluate(sloppy_student, data, max_examples=10, previous=previous)

    reg = metrics["regression"]
    assert reg["previous"] == previous
    # tool_name_accuracy dropped from 0.95 → 0.333 → a real regression
    assert "tool_name_accuracy" in reg["deltas"]
    assert reg["deltas"]["tool_name_accuracy"] < 0
    # And the human-readable list flags at least the tool-name regression
    assert any("tool_name_accuracy" in r for r in reg["regressions"])


def test_evaluate_with_previous_metrics_no_regression_when_better():
    from distillarium.tasting import evaluate

    data = _eval_data()
    perfect_student = _DictDrivenGenerator({
        "text mom hi":            [{"name": "send_message", "args": {"contact": "mom"}}],
        "start a 5 minute timer": [{"name": "set_timer",    "args": {"minutes": 5}}],
        "text dad about dinner":  [{"name": "send_message", "args": {"contact": "dad"}}],
    })

    previous = {"tool_name_accuracy": 0.5, "exact_call_accuracy": 0.5}
    metrics = evaluate(perfect_student, data, max_examples=10, previous=previous)

    reg = metrics["regression"]
    # Strict improvements — no regressions
    assert reg["regressions"] == []
    assert reg["deltas"]["tool_name_accuracy"] == 0.5
    assert reg["deltas"]["exact_call_accuracy"] == 0.5


def test_evaluate_loads_previous_from_json_file(tmp_path: Path):
    from distillarium.tasting import evaluate

    data = _eval_data()
    perfect_student = _DictDrivenGenerator({
        "text mom hi":            [{"name": "send_message", "args": {"contact": "mom"}}],
        "start a 5 minute timer": [{"name": "set_timer",    "args": {"minutes": 5}}],
        "text dad about dinner":  [{"name": "send_message", "args": {"contact": "dad"}}],
    })

    prev_path = tmp_path / "prev.json"
    prev_path.write_text(json.dumps({"tool_name_accuracy": 0.6}))

    metrics = evaluate(perfect_student, data, max_examples=10, previous=str(prev_path))
    assert metrics["regression"]["previous"]["tool_name_accuracy"] == 0.6


def test_evaluate_loads_previous_from_tasting_notes_payload(tmp_path: Path):
    """Accept a payload nested under 'student' or 'metrics' for convenience."""
    from distillarium.tasting import evaluate

    data = _eval_data()
    perfect_student = _DictDrivenGenerator({
        "text mom hi":            [{"name": "send_message", "args": {"contact": "mom"}}],
        "start a 5 minute timer": [{"name": "set_timer",    "args": {"minutes": 5}}],
        "text dad about dinner":  [{"name": "send_message", "args": {"contact": "dad"}}],
    })

    prev_path = tmp_path / "notes.json"
    prev_path.write_text(json.dumps({
        "student": {"tool_name_accuracy": 0.7},
        "samples": [],  # Tasting Notes-shaped payload
    }))

    metrics = evaluate(perfect_student, data, max_examples=10, previous=str(prev_path))
    assert metrics["regression"]["previous"] == {"tool_name_accuracy": 0.7}
