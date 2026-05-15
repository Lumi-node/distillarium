"""Tasting — held-out evaluation + Tasting Notes generation."""

from distillarium.tasting.evaluator import (
    evaluate,
    parse_generated_call,
    TeacherEvalGenerator,
)

__all__ = ["evaluate", "parse_generated_call", "TeacherEvalGenerator"]
