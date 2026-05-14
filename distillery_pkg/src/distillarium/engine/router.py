from __future__ import annotations

from distillarium.engine.inference import FunctionCallGenerator
from distillarium.engine.schema import SchemaEncoder


class FunctionCallRouter:
    def __init__(
        self,
        generator: FunctionCallGenerator,
        schema_encoder: SchemaEncoder,
    ):
        self.generator = generator
        self.schema_encoder = schema_encoder

    def route(
        self, utterance: str, tool_definitions: list[dict]
    ) -> list[dict]:
        return self.generator.generate(utterance, tool_definitions)
