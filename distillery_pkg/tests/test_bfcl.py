"""Tests for the BFCL alignment helpers — no network, fixture is in-test."""

from __future__ import annotations

import json
from pathlib import Path


# A 2-row BFCL-shaped fixture covering one `simple` example (single tool, single
# call) and one `parallel` example (multiple gold calls in one utterance).
_FIXTURE = [
    {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "Text mom hi"}]],
        "function": [{
            "name": "send_message",
            "description": "Send a text message.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "contact": {"type": "string", "description": "Recipient"},
                    "message": {"type": "string", "description": "Body"},
                },
                "required": ["contact", "message"],
            },
        }],
        "ground_truth": [{"send_message": {"contact": "mom", "message": "hi"}}],
    },
    {
        "id": "parallel_0",
        "question": [[{"role": "user", "content": "Text dad about dinner and set a 5 min timer"}]],
        "function": [
            {
                "name": "send_message",
                "parameters": {"type": "dict",
                               "properties": {"contact": {"type": "string"},
                                              "message": {"type": "string"}},
                               "required": ["contact", "message"]},
            },
            {
                "name": "set_timer",
                "parameters": {"type": "dict",
                               "properties": {"minutes": {"type": "integer"}},
                               "required": ["minutes"]},
            },
        ],
        "ground_truth": [
            {"send_message": {"contact": "dad", "message": "dinner"}},
            {"set_timer": {"minutes": 5}},
        ],
    },
]


def _write_fixture(path: Path) -> None:
    path.write_text("\n".join(json.dumps(row) for row in _FIXTURE))


def test_load_bfcl_split_yields_3_tuples(tmp_path: Path):
    from distillarium.tasting import load_bfcl_split

    p = tmp_path / "BFCL_v3_synth.json"
    _write_fixture(p)

    rows = list(load_bfcl_split(p))
    assert len(rows) == 2

    utt0, tools0, gold0 = rows[0]
    assert utt0 == "Text mom hi"
    assert len(tools0) == 1
    assert tools0[0]["name"] == "send_message"
    assert tools0[0]["params"]["contact"]["required"] is True
    gold0_parsed = json.loads(gold0)
    assert gold0_parsed[0] == {"name": "send_message",
                               "args": {"contact": "mom", "message": "hi"}}


def test_load_bfcl_split_handles_parallel_multi_call(tmp_path: Path):
    from distillarium.tasting import load_bfcl_split

    p = tmp_path / "BFCL_v3_synth.json"
    _write_fixture(p)
    rows = list(load_bfcl_split(p))

    _, _, gold1 = rows[1]
    gold1_parsed = json.loads(gold1)
    assert len(gold1_parsed) == 2
    assert gold1_parsed[0]["name"] == "send_message"
    assert gold1_parsed[1]["name"] == "set_timer"


def test_score_against_bfcl_scores_a_generator(tmp_path: Path):
    from distillarium.tasting import score_against_bfcl

    p = tmp_path / "BFCL_v3_synth.json"
    _write_fixture(p)

    class _PerfectGen:
        def generate(self, utt, tools):
            # Return what gold expects for the first call of either example
            if "mom" in utt:
                return [{"name": "send_message",
                         "args": {"contact": "mom", "message": "hi"}}]
            return [{"name": "send_message",
                     "args": {"contact": "dad", "message": "dinner"}}]

    metrics = score_against_bfcl(_PerfectGen(), p, max_examples=10)
    assert metrics["bfcl_split"] == "BFCL_v3_synth.json"
    assert metrics["tool_name_accuracy"] == 1.0  # both got the FIRST call right


def test_load_bfcl_split_raises_on_missing_file(tmp_path: Path):
    import pytest
    from distillarium.tasting import load_bfcl_split

    with pytest.raises(FileNotFoundError, match="BFCL split not found"):
        list(load_bfcl_split(tmp_path / "does-not-exist.json"))
