# ⚗ The Distillery

> Distill any task into a pocket-sized Spirit. Pure model. No API.

`distillarium` is a Python package that turns a teacher LLM (Gemini, Claude, GPT) into a tiny, deployable, task-specific model — a **Spirit** — that runs on CPU, edge, or browser with zero API dependency at inference.

## Why distillation, not fine-tuning

| | LoRA / fine-tuning | **Distillation (The Distillery)** |
|---|---|---|
| Final size | 7B+ (same as base) | 5M–50M |
| Inference target | GPU | CPU, edge, browser |
| API dependency at inference | Sometimes (for hosted base) | **Zero** |
| Cost at inference | $$$ per call | $0 |
| Cold start | Seconds | Milliseconds |
| Best for | Capability extension on a generalist | Single-task production |

## Install

```bash
pip install distillarium[gemini]
```

## Hello, Spirit

Three commands, one Spirit:

```bash
# 1. Distill a teacher into a Spirit using a recipe
distillery distill recipes/needle.tool-calling-v1.yaml

# 2. Taste the Spirit — held-out eval against the teacher
distillery taste spirits/needle.pt

# 3. Bottle the Spirit — export to ONNX / GGUF for deployment
distillery bottle spirits/needle.pt --format onnx
```

## A real example — Needle (tool calling, 20.7M params)

Reproduces today's reference Spirit at 67° proof on tool-calling:

```yaml
# recipes/needle.tool-calling-v1.yaml
name: needle.tool-calling
version: 1

teacher:
  provider: gemini
  model: gemini-2.5-flash
  temperature: 0.9

mash:
  total_examples: 1000
  examples_per_call: 10
  tools_per_call: { min: 3, max: 6 }

student:
  arch: attention-only-glu
  d_model: 384
  n_heads: 6
  n_layers: 8
  max_seq_len: 256
  tokenizer: wordpiece-4096

cuts: { train: 0.9, eval: 0.1 }

still:
  epochs: 8
  batch_size: 16
  lr: 3.0e-4

tasting:
  metrics: [tool_name_accuracy, arg_key_f1, exact_call_accuracy]
  held_out: 100
```

Run it:
```bash
distillery distill recipes/needle.tool-calling-v1.yaml --out spirits/
```

Expected result on a single RTX 5090: **~30 minutes, ~$0.30 in Gemini Flash API**, 67% tool-name accuracy on held-out.

## The Distillation Vocabulary

| Term | Means |
|---|---|
| **Spirit** | The trained, bottled model (your output) |
| **Mash** | Seed corpus the teacher generates training data from |
| **Recipe** | YAML config — teacher, mash, student arch, cuts, still, tasting, bottling |
| **The Still** | The training run |
| **Cuts** | Train / eval / test data splits |
| **Heads / Hearts / Tails** | Discarded noise / kept core / borderline cases |
| **Proof** | Held-out accuracy. The higher the proof, the more concentrated. |
| **Tasting Notes** | Auto-generated eval report with strengths, weaknesses, failure cases |
| **Aging in Casks** | Continued training, fine-tuning, RLHF refresh |
| **Bottling** | Export to ONNX / GGUF / browser-WASM |
| **The Cellar** | Library of Spirits (public or private) |

## Python API

```python
from distillarium import distill, taste, bottle, Recipe

# Load a recipe
recipe = Recipe.from_file("recipes/needle.tool-calling-v1.yaml")

# Distill
spirit = distill(recipe)

# Taste (eval against held-out cuts)
notes = taste(spirit, held_out=100)
print(notes.metrics)
# {'tool_name_accuracy': 0.67, 'arg_key_f1': 0.69, ...}

# Bottle (export)
bottle(spirit, format="onnx", out="spirits/needle.onnx")
```

## What's in the box

- `distillarium.engine` — attention-only transformer architecture + tokenizer + trainer + inference
- `distillarium.teacher` — pluggable teacher backends (Gemini, more coming)
- `distillarium.tasting` — held-out evaluation + Tasting Notes generation
- `distillarium.bottling` — exporters (ONNX in v0.1, GGUF in v0.2)
- `distillarium.cli` — `distillery distill | taste | bottle` commands

## Status

- ✅ v0.1 — Tool-calling Spirits via Gemini (this release)
- 🚧 v0.2 — Claude teacher, GGUF export, byte-level BPE tokenizer
- 🚧 v0.3 — Classification Spirits, RAG-routing Spirits
- 🚧 v0.4 — Quantization-aware distillation

## License

MIT.

---

*Built on top of the [Research Radar](https://github.com/the-distillery/research-radar) pipeline. First reference Spirit ([Needle](https://thedistillery.run/cellar/needle)) distilled 2026-05-13.*
