"""Recipe — YAML schema for a distillation run.

A Recipe captures every knob: teacher, mash, student arch, cuts, still, tasting,
bottling. It's the durable, forkable, versionable artifact users iterate on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TeacherSpec:
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.9
    api_key: str | None = None  # Falls back to env GOOGLE_API_KEY


@dataclass
class MashSpec:
    total_examples: int = 1000
    examples_per_call: int = 10
    tools_per_call: dict = field(default_factory=lambda: {"min": 3, "max": 6})
    seed: int = 2026


@dataclass
class StudentSpec:
    arch: str = "attention-only-glu"
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 8
    max_seq_len: int = 256
    tokenizer: str = "wordpiece-4096"


@dataclass
class CutsSpec:
    train: float = 0.9
    eval: float = 0.1


@dataclass
class StillSpec:
    epochs: int = 8
    batch_size: int = 16
    lr: float = 3.0e-4
    optimizer: str = "adamw"
    grad_clip: float = 1.0


@dataclass
class TastingSpec:
    metrics: list[str] = field(
        default_factory=lambda: [
            "tool_name_accuracy",
            "arg_key_f1",
            "exact_call_accuracy",
        ]
    )
    held_out: int = 100
    decode: dict = field(default_factory=lambda: {"temp": 0.1, "json_mask": True})


@dataclass
class BottlingSpec:
    formats: list[str] = field(default_factory=lambda: ["pytorch"])
    signed: bool = False


@dataclass
class Recipe:
    """A complete distillation specification."""
    name: str
    version: int = 1
    teacher: TeacherSpec = field(default_factory=TeacherSpec)
    mash: MashSpec = field(default_factory=MashSpec)
    student: StudentSpec = field(default_factory=StudentSpec)
    cuts: CutsSpec = field(default_factory=CutsSpec)
    still: StillSpec = field(default_factory=StillSpec)
    tasting: TastingSpec = field(default_factory=TastingSpec)
    bottling: BottlingSpec = field(default_factory=BottlingSpec)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Recipe":
        def _sub(key: str, klass):
            sub = d.get(key, {})
            if isinstance(sub, klass):
                return sub
            return klass(**(sub or {}))

        return cls(
            name=d["name"],
            version=d.get("version", 1),
            teacher=_sub("teacher", TeacherSpec),
            mash=_sub("mash", MashSpec),
            student=_sub("student", StudentSpec),
            cuts=_sub("cuts", CutsSpec),
            still=_sub("still", StillSpec),
            tasting=_sub("tasting", TastingSpec),
            bottling=_sub("bottling", BottlingSpec),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Recipe":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "teacher": self.teacher.__dict__,
            "mash": self.mash.__dict__,
            "student": self.student.__dict__,
            "cuts": self.cuts.__dict__,
            "still": self.still.__dict__,
            "tasting": self.tasting.__dict__,
            "bottling": self.bottling.__dict__,
        }
