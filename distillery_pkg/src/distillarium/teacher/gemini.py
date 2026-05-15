"""Gemini teacher backend.

Generates (utterance, available_tools, target_call) triples by prompting Gemini
to produce realistic natural-language queries that map to specific tool calls.

Cost target: <$1 for 1K examples using gemini-2.5-flash (batched generation).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Iterator

from distillarium.teacher.base import Teacher, DistillExample


# Curated tool catalog. The model sees a subset per generation call.
TOOL_CATALOG: list[dict] = [
    {
        "name": "send_message",
        "description": "Send a text message to a contact.",
        "params": {
            "contact": {"type": "string", "required": True, "description": "Recipient name"},
            "message": {"type": "string", "required": True, "description": "Message body"},
        },
    },
    {
        "name": "set_timer",
        "description": "Start a countdown timer.",
        "params": {
            "minutes": {"type": "integer", "required": True},
            "seconds": {"type": "integer", "required": False, "description": "Optional seconds"},
        },
    },
    {
        "name": "create_event",
        "description": "Create a calendar event.",
        "params": {
            "title": {"type": "string", "required": True},
            "date": {"type": "string", "required": True, "description": "ISO date or natural"},
            "time": {"type": "string", "required": False},
            "participants": {"type": "array", "required": False},
        },
    },
    {
        "name": "get_weather",
        "description": "Look up the weather forecast.",
        "params": {
            "location": {"type": "string", "required": True},
            "day": {"type": "string", "required": False, "description": "today, tomorrow, etc."},
        },
    },
    {
        "name": "play_music",
        "description": "Play a song, artist, or playlist.",
        "params": {
            "query": {"type": "string", "required": True},
            "shuffle": {"type": "boolean", "required": False},
        },
    },
    {
        "name": "navigate_to",
        "description": "Start turn-by-turn navigation.",
        "params": {
            "destination": {"type": "string", "required": True},
            "mode": {"type": "string", "required": False, "description": "driving, walking, transit"},
        },
    },
    {
        "name": "set_reminder",
        "description": "Set a future reminder.",
        "params": {
            "text": {"type": "string", "required": True},
            "when": {"type": "string", "required": True},
        },
    },
    {
        "name": "send_email",
        "description": "Send an email.",
        "params": {
            "to": {"type": "string", "required": True},
            "subject": {"type": "string", "required": True},
            "body": {"type": "string", "required": True},
        },
    },
    {
        "name": "search_web",
        "description": "Run a web search.",
        "params": {"query": {"type": "string", "required": True}},
    },
    {
        "name": "control_device",
        "description": "Control a smart home device.",
        "params": {
            "device": {"type": "string", "required": True},
            "action": {"type": "string", "required": True, "description": "on, off, set, etc."},
            "value": {"type": "string", "required": False},
        },
    },
    {
        "name": "transfer_money",
        "description": "Transfer money between accounts.",
        "params": {
            "amount": {"type": "number", "required": True},
            "from_account": {"type": "string", "required": True},
            "to_account": {"type": "string", "required": True},
        },
    },
    {
        "name": "log_health_metric",
        "description": "Log a health metric (weight, steps, etc.).",
        "params": {
            "metric": {"type": "string", "required": True},
            "value": {"type": "number", "required": True},
            "unit": {"type": "string", "required": False},
        },
    },
    {
        "name": "post_social",
        "description": "Post to a social media platform.",
        "params": {
            "platform": {"type": "string", "required": True},
            "content": {"type": "string", "required": True},
        },
    },
    {
        "name": "order_item",
        "description": "Order an item from a store.",
        "params": {
            "item": {"type": "string", "required": True},
            "quantity": {"type": "integer", "required": False},
        },
    },
    {
        "name": "translate_text",
        "description": "Translate text between languages.",
        "params": {
            "text": {"type": "string", "required": True},
            "target_language": {"type": "string", "required": True},
        },
    },
]


SYSTEM_PROMPT = """\
You are a data generator for training a small function-calling model. Given a set of
available tools, produce realistic and DIVERSE natural-language user utterances and
the correct function call(s) for each.

Rules:
1. Each utterance must be something a user would actually say (vary phrasing,
   formality, length, and include occasional small typos or filler words).
2. The target call must be a JSON list of {"name": str, "args": dict} objects.
3. About 80% of examples should call ONE tool; 20% should chain TWO tools.
4. Args must match the tool's required parameter names exactly.
5. NEVER produce an utterance that needs a tool not in the available set.
6. Output ONLY valid JSON in the exact schema requested. No commentary.

Examples are for distillation — diversity matters more than perfection. Cover
imperatives ("Set a timer..."), questions ("What's the weather..."), and
declaratives ("I need to message Mom...").
"""

USER_PROMPT_TEMPLATE = """\
Available tools (this exact set is what the model will see):
{tools_json}

Produce {n} training examples as a JSON object with one top-level key "examples"
whose value is an array of {n} objects, each:
{{
  "utterance": "<user utterance>",
  "target_call": [{{"name": "<tool_name>", "args": {{...}}}}, ...]
}}

Make them DIVERSE. Vary contact names, locations, devices, items, times, etc.
Some examples should be slightly ambiguous so the model learns to pick the right
tool from context.
"""


# ---- Inference-time prompt (used by TeacherEvalGenerator for the teacher
# baseline in evaluate()). Different from data-generation prompts above:
# here the teacher is the "student" — given ONE utterance and tools, pick
# the right call. ---------------------------------------------------------
EVAL_SYSTEM_PROMPT = """\
You are a function-caller. Given a user utterance and the available tools,
return the SINGLE best tool call as JSON. Output ONLY valid JSON — no prose,
no code fences, no commentary.

Schema:
{"name": "<tool_name>", "args": {"<arg_key>": "<arg_value>", ...}}

Rules:
1. Use only tool names from the provided set.
2. Use only argument keys defined in that tool's params.
3. Required args must be present; optional args may be omitted.
4. If no tool fits, return {"name": "", "args": {}}.
"""

EVAL_USER_PROMPT_TEMPLATE = """\
Available tools:
{tools_json}

User utterance: {utterance}

Return the JSON call only.
"""


class GeminiTeacher(Teacher):
    """Gemini-Flash-backed teacher. Yields realistic (utt, tools, target_call) triples."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        examples_per_call: int = 10,
        tools_per_call_min: int = 3,
        tools_per_call_max: int = 6,
        seed: int = 0,
        tool_catalog: list[dict] | None = None,
    ):
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise RuntimeError("google-genai not installed. Run: pip install google-genai") from e

        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model = model
        self.examples_per_call = examples_per_call
        self.tmin = tools_per_call_min
        self.tmax = tools_per_call_max
        self.rng = random.Random(seed)
        self.failures = 0
        self.tool_catalog = tool_catalog if tool_catalog is not None else TOOL_CATALOG

    def _pick_tools(self, must_include: dict | None = None) -> list[dict]:
        n = self.rng.randint(self.tmin, self.tmax)
        pool = [t for t in self.tool_catalog if t is not must_include]
        sampled = self.rng.sample(pool, k=min(n - (1 if must_include else 0), len(pool)))
        out = ([must_include] if must_include else []) + sampled
        self.rng.shuffle(out)
        return out

    def _parse_response(
        self, text: str, tools: list[dict]
    ) -> list[DistillExample]:
        out: list[DistillExample] = []
        # Strip code fences if present
        s = text.strip()
        if s.startswith("```"):
            # ```json\n...\n```
            s = s.split("\n", 1)[1] if "\n" in s else s
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0]
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            self.failures += 1
            return out

        examples = data.get("examples", []) if isinstance(data, dict) else []
        valid_tool_names = {t["name"] for t in tools}

        for ex in examples:
            utt = (ex or {}).get("utterance")
            calls = (ex or {}).get("target_call")
            if not isinstance(utt, str) or not utt.strip():
                continue
            if not isinstance(calls, list) or not calls:
                continue
            ok = True
            for c in calls:
                if not isinstance(c, dict):
                    ok = False
                    break
                if c.get("name") not in valid_tool_names:
                    ok = False
                    break
                if not isinstance(c.get("args", {}), dict):
                    ok = False
                    break
            if not ok:
                continue
            out.append(DistillExample(utterance=utt.strip(), tools=tools, target_call=calls))
        return out

    def answer(self, utterance: str, tools: list[dict]) -> list[dict]:
        """Inference-time call against the teacher — used for teacher-baseline eval."""
        user_prompt = EVAL_USER_PROMPT_TEMPLATE.format(
            tools_json=json.dumps(tools, indent=2),
            utterance=utterance,
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[EVAL_SYSTEM_PROMPT, user_prompt],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.0,  # deterministic at inference time
                    "max_output_tokens": 512,
                },
            )
            text = (resp.text or "").strip()
        except Exception:
            self.failures += 1
            return []

        # Strip code fences if any model variant ignored response_mime_type
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]

        try:
            call = json.loads(text)
        except json.JSONDecodeError:
            return []

        # Accept either {"name", "args"} or a list of such objects (defensive)
        if isinstance(call, list):
            call = call[0] if call else {}
        if not isinstance(call, dict):
            return []
        name = call.get("name", "")
        args = call.get("args", {})
        if not isinstance(name, str) or not name:
            return []
        if not isinstance(args, dict):
            args = {}
        return [{"name": name, "args": args}]

    def generate_batch(self, batches: int, verbose: bool = True) -> Iterator[DistillExample]:
        for b in range(batches):
            tools = self._pick_tools()
            user_prompt = USER_PROMPT_TEMPLATE.format(
                tools_json=json.dumps(tools, indent=2),
                n=self.examples_per_call,
            )
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=[SYSTEM_PROMPT, user_prompt],
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.9,  # high temp for diversity
                        "max_output_tokens": 4096,
                    },
                )
                text = resp.text or ""
            except Exception as e:
                if verbose:
                    print(f"  [batch {b+1}/{batches}] API error: {e}", flush=True)
                self.failures += 1
                continue

            examples = self._parse_response(text, tools)
            if verbose:
                print(
                    f"  [batch {b+1}/{batches}] {len(examples)}/{self.examples_per_call} valid "
                    f"(tools={len(tools)})",
                    flush=True,
                )
            for ex in examples:
                yield ex
