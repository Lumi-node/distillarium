# ⚗ Distillarium

> **Replace narrow LLM pipeline steps with tiny CPU-runnable Spirits.**
> Distill a teacher API into a deployable 20–50M-param model. Same outputs, $0 at inference.

`distillarium` is a Python toolkit for distilling a teacher LLM (Gemini, Claude, GPT) into a tiny, task-specific model — a **Spirit** — that runs on CPU, edge, or browser with zero API dependency at inference.

**v0.1.1 — alpha.** The reference Spirit (Needle, tool calling) is published; the toolkit itself is single-task at the moment with more recipes coming. Pipeline-replacement claims in the docs are based on Needle's measured numbers (78% tool-name accuracy on held-out cuts). See *Status* below.

> **Pick this if:** you're doing the same narrow LLM call thousands of times a day (intent, routing, NER, classification, tool calling) and want to stop paying frontier-API prices for it.
> **Don't pick this if:** you need open-ended generation, multi-step reasoning, or a generalist assistant — keep the frontier model for that.

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

Reproduces the reference Spirit at **78° proof** on tool-name accuracy
(held-out cuts of 100 examples):

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

Expected result on a single RTX 5090: **~30 minutes, ~$0.30 in Gemini Flash API**, **78% tool-name accuracy** on held-out, **0.73 arg-key F1**, and **3% exact-call accuracy** (the value-prediction weak spot we're working on — see *Status*).

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
# {'tool_name_accuracy': 0.78, 'arg_key_f1': 0.73, 'exact_call_accuracy': 0.03, ...}

# Bottle (export)
bottle(spirit, format="onnx", out="spirits/needle.onnx")
```

## What's in the box

- `distillarium.engine` — attention-only transformer architecture + tokenizer + trainer + inference
- `distillarium.teacher` — pluggable teacher backends (Gemini, more coming)
- `distillarium.tasting` — held-out evaluation + Tasting Notes generation
- `distillarium.bottling` — exporters (ONNX in v0.1, GGUF in v0.2)
- `distillarium.cli` — `distillery distill | taste | bottle` commands

## How it compares

This sits in a specific gap in the existing distillation ecosystem:

| Project | Sweet spot | Where Distillarium differs |
|---|---|---|
| [Arcee DistillKit](https://github.com/arcee-ai/DistillKit) | General LLM distillation pipelines, 7B-target | We target the **5–50M class**, deployment-first (.onnx/.gguf/.wasm as the output, not an afterthought) |
| [ModelScope EasyDistill](https://github.com/modelscope/EasyDistill) (incl. AgentKD) | Agent distillation, multi-modal | We're CPU/edge-deployment focused, single CLI, Tasting Notes as default eval rigor |
| [Berkeley TinyAgent](https://github.com/SqueezeAILab/TinyAgent) | 1.1B–7B function-calling SLMs | We go smaller (20–50M) at the cost of generality; gain CPU inference and zero vendor lock |
| LoRA fine-tuning | Capability extension on a generalist | Doesn't shrink the model. Distillarium produces a small *student* model, not a fine-tuned base |

**Honest framing:** for breadth and SOTA function-calling scores, TinyAgent is the right pick. For replacing one narrow LLM step in a production pipeline with a CPU-runnable artifact you can audit, fork, and ship — that's what Distillarium is for.

We plan to publish BFCL ([Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)) numbers for Needle in v0.2 so the comparison is apples-to-apples.

## Status

**Current release: v0.1.1 (alpha)** — *single-task: tool calling via Gemini Flash. The reference Needle Spirit is published; other Spirits listed on the site are roadmap items.*

| Version | Scope | Status |
|---|---|---|
| **v0.1.1** | Tool-calling Spirits via Gemini · Needle published · pixel-art metaphor UX · ONNX bottling stub | **Shipped** ✅ |
| v0.2 | Claude + OpenAI teacher backends · byte-level BPE tokenizer (fixes argument-value accuracy) · GGUF export · BFCL benchmark numbers · `taste` shows teacher-vs-student baseline + version regression | In progress |
| v0.3 | Classification Spirits · NER Spirits · RAG-routing Spirits · iterative re-distillation on failed cuts | Planned |
| v0.4 | Quantization-aware training · WebAssembly bottling | Planned |

### What's NOT solved in v0.1.1 (be honest)

- **Argument-value accuracy.** Exact-call sits at 3% on Needle. The WordPiece tokenizer splits JSON values awkwardly. v0.2's byte-level BPE is the fix.
- **Only Gemini teacher** is wired up. Claude and OpenAI providers are stubs.
- **No BFCL score yet.** v0.2 will publish.
- **Tasting Notes are statistical.** They don't catch semantic failures like predicting "Twitter" instead of "Instagram." LLM-as-judge eval is on the roadmap.

## License

MIT.

---

*Built on top of the [Research Radar](https://distillarium.app#research-radar) — [Automate Capture's](https://automate-capture.com) autonomous research-to-product pipeline. The reference Needle Spirit was distilled from a Show HN paper the Radar surfaced in May 2026.*
