"""Abstract teacher interface — every provider implements `generate_batch`."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


@dataclass
class DistillExample:
    """One (utterance, tools, target_call) training triple."""
    utterance: str
    tools: list[dict]
    target_call: list[dict]  # list of {name, args}

    def as_train_tuple(self) -> tuple[str, list[dict], str]:
        return (self.utterance, self.tools, json.dumps(self.target_call))

    def to_dict(self) -> dict:
        return asdict(self)


class Teacher(ABC):
    """Base class for all teacher backends."""

    @abstractmethod
    def generate_batch(
        self, batches: int, verbose: bool = True
    ) -> Iterator[DistillExample]:
        """Yield DistillExample objects, one per valid teacher output."""
        raise NotImplementedError

    def answer(self, utterance: str, tools: list[dict]) -> list[dict]:
        """Inference-time call: given one utterance + tool set, pick the call.

        Used by `TeacherEvalGenerator` to compute the teacher baseline during
        Tasting. Returns `[{"name": ..., "args": ...}]` (a list to match the
        student generator's shape), or `[]` if no valid tool fits.

        The default implementation raises — subclasses opt in by overriding.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement inference-time answer(). "
            "Override answer() on this teacher to enable teacher-baseline eval."
        )

    def distill_to_file(
        self,
        out_path: str | Path,
        n_examples: int,
        examples_per_call: int = 10,
    ) -> dict:
        """Run the teacher until `n_examples` valid examples are written.

        Returns a stats dict: {out_path, examples_written, api_failures}.
        """
        import time

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Generous upper bound on batches (caller should override for tight runs)
        batches_needed = max(1, (n_examples * 12 // max(1, examples_per_call)))
        t0 = time.time()
        count = 0
        with out_path.open("w") as f:
            for ex in self.generate_batch(batches_needed):
                f.write(json.dumps(ex.to_dict()) + "\n")
                count += 1
                if count >= n_examples:
                    break

        elapsed = time.time() - t0
        return {
            "out_path": str(out_path),
            "examples_written": count,
            "elapsed_s": round(elapsed, 1),
            "api_failures": getattr(self, "failures", 0),
        }


def load_distilled(path: str | Path) -> list[tuple[str, list[dict], str]]:
    """Load a distilled JSONL file as (utterance, tools, target_json_str) tuples."""
    out: list[tuple[str, list[dict], str]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append((row["utterance"], row["tools"], json.dumps(row["target_call"])))
    return out
