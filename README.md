<!--
  This is the top-level README that will land at
  github.com/the-distillery/distillarium/README.md

  Move it to the root of the public repo (above distillery_pkg/, site/, etc.)
  when you push.
-->

<h1 align="center">
  <img src="site/public/favicon.svg" width="64" height="64" alt="alembic" /><br/>
  Distillarium
</h1>

<p align="center">
  <strong>Distill any task into a pocket-sized Spirit. Pure model. No API.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/distillarium/"><img src="https://img.shields.io/pypi/v/distillarium.svg?color=e8c547&label=pypi" alt="PyPI"/></a>
  <a href="https://pypi.org/project/distillarium/"><img src="https://img.shields.io/pypi/pyversions/distillarium.svg?color=ffa432" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-6abe30.svg" alt="MIT"/></a>
  <a href="https://distillarium.app"><img src="https://img.shields.io/badge/site-distillarium.app-c97d3b.svg" alt="Site"/></a>
</p>

<p align="center">
  <a href="https://distillarium.app">Site</a> ·
  <a href="https://distillarium.app/cellar">The Cellar</a> ·
  <a href="https://distillarium.app/docs">Docs</a> ·
  <a href="https://distillarium.app/cellar/needle">Reference Spirit: Needle</a>
</p>

---

## What it is

`distillarium` is an open-source Python toolkit that **distills a teacher LLM** (Gemini, Claude, GPT) **into a tiny, deployable, task-specific model** — what we call a **Spirit**. Spirits run on CPU, edge, or browser with **zero API dependency at inference**.

Three commands:

```bash
pip install distillarium

distillery distill recipes/needle.tool-calling-v1.yaml
distillery bottle spirits/needle.pt --format onnx
```

That's it. You hand it a teacher API key and a task spec; it hands back a 20–50M parameter model that runs anywhere.

## Why distillation, not fine-tuning

| | LoRA / fine-tuning | **Distillation (distillarium)** |
|---|---|---|
| Final model size | 7B+ (same as base) | **5M–50M** |
| Inference target | GPU | **CPU, edge, browser** |
| API dependency at inference | Sometimes (for hosted base) | **Zero** |
| Cost at inference | $$$ per call | **$0** |
| Cold start | Seconds | **Milliseconds** |
| Best for | Generalist capability extension | **Single-task production** |

## A real example: Needle

The reference Spirit shipped at v0.1 — an attention-only 20.7M transformer that does function calling.

| | Value |
|---|---|
| Teacher | `gemini-2.5-flash` |
| Compression | ~72,000× (1.5T → 20.7M params) |
| Distillation cost | **$0.30** (Gemini Flash) |
| Distillation time | **27 minutes** on a single RTX 5090 |
| Final loss | 0.71 (over 8 epochs) |
| Tool-name accuracy | **78%** on held-out cuts |
| Arg-key F1 | 0.73 (p=0.85, r=0.64) |
| Final size on disk | 249 MB (PyTorch) |
| Inference target | CPU, ~45ms median latency |

📜 [Full Tasting Notes for Needle →](https://distillarium.app/cellar/needle)

## The distillation vocabulary

Every concept in `distillarium` maps to something real in distilling.

| Term | What it means in ML |
|---|---|
| **Spirit** | The trained, bottled model (the output) |
| **Mash** | Seed corpus the teacher generates training data from |
| **Recipe** | YAML config — teacher, mash, student arch, cuts, still, tasting, bottling |
| **The Still** | The training run |
| **Cuts** | Train / eval / test data splits |
| **Heads / Hearts / Tails** | Discarded noise / kept core / borderline cases |
| **Proof** | Held-out accuracy. The higher the proof, the more concentrated. |
| **Tasting Notes** | Auto-generated eval report — strengths, weaknesses, failure cases |
| **Aging in Casks** | Continued training, fine-tuning, RLHF refresh |
| **Bottling** | Export to ONNX, GGUF, or browser-WASM |
| **The Cellar** | Library of Spirits (public or private) |

## Quick start

```bash
# 1. Install
pip install distillarium[gemini]

# 2. Set a teacher key (any provider — Gemini Flash is cheapest)
export GOOGLE_API_KEY=your-key

# 3. Distill — uses the reference Needle recipe
distillery distill recipes/needle.tool-calling-v1.yaml

# 4. Inspect your local Cellar
distillery cellar

# 5. Re-taste against fresh held-out data
distillery taste spirits/needle.pt --mash held_out.jsonl

# 6. Bottle for deployment (ONNX, GGUF, WASM)
distillery bottle spirits/needle.pt --format onnx
```

Expected on a single GPU: ~30 minutes, ~$0.30 in teacher API, 70–80° proof on tool calling.

## Python API

```python
from distillarium import Recipe, distill, taste, bottle

# Load a recipe
recipe = Recipe.from_file("recipes/needle.tool-calling-v1.yaml")

# Distill
spirit = distill(recipe)

# Taste (held-out eval)
notes = taste(spirit, eval_data_path="held_out.jsonl", held_out=100)
print(notes)
# {'tool_name_accuracy': 0.78, 'arg_key_f1': 0.73, ...}

# Bottle (export)
bottle(spirit, format="onnx", out="spirits/needle.onnx")
```

## Repo structure

```
distillarium/
├── distillery_pkg/             ← the Python package (published as `distillarium`)
│   ├── pyproject.toml
│   ├── src/distillarium/
│   │   ├── engine/             ← attention-only transformer + tokenizer + trainer
│   │   ├── teacher/            ← pluggable teacher backends (Gemini, more coming)
│   │   ├── tasting/            ← evaluator + Tasting Notes generator
│   │   ├── bottling/           ← exporters (ONNX in v0.1, GGUF in v0.2)
│   │   ├── cli.py              ← `distillery distill | taste | bottle` commands
│   │   └── pipeline.py         ← high-level distill() / taste() / bottle()
│   ├── recipes/                ← reference recipes
│   ├── tests/
│   └── release.sh              ← local PyPI release helper
│
├── site/                       ← Astro 5 site (distillarium.app)
│   ├── src/
│   │   ├── layouts/Base.astro
│   │   ├── components/         ← StillCanvas, BottleCard
│   │   ├── data/cellar.json    ← static catalog of public Spirits
│   │   ├── pages/
│   │   │   ├── index.astro     ← landing page with live distillation animation
│   │   │   ├── cellar/         ← bottle grid + dynamic spirit detail pages
│   │   │   └── docs.astro
│   │   └── styles/global.css
│   └── wrangler.toml           ← Cloudflare Pages config
│
├── recipes/                    ← community recipe library (forkable)
│
├── .github/workflows/
│   ├── deploy-site.yml         ← auto-deploys site/ to Cloudflare Pages on push
│   └── publish-pypi.yml        ← auto-publishes package on `v*.*.*` tag
│
├── DEPLOY.md                   ← end-to-end deployment playbook
├── LICENSE                     ← MIT
└── README.md                   ← you are here
```

## The Cellar — featured Spirits

| Spirit | Task | Teacher | Params | Proof | Size |
|---|---|---|---|---|---|
| [Needle](https://distillarium.app/cellar/needle) | Tool calling | Gemini 2.5 Flash | 20.7M | 78° | 249 MB |
| [PII Guard](https://distillarium.app/cellar/pii-guard) | Privacy / compliance | Claude Sonnet 4.6 | 14M | 82° | 56 MB |
| [Claimant](https://distillarium.app/cellar/claimant) | Fact checking | Gemini 2.5 Pro | 32M | 91° | 128 MB |
| [Routor](https://distillarium.app/cellar/routor) | Intent classification | Claude Sonnet 4.6 | 8M | 74° | 32 MB |

Click any Spirit for full Tasting Notes, the Recipe that produced it, and download links.

## Contributing a Spirit to the public Cellar

1. Fork this repo
2. Write your recipe at `recipes/<your-spirit-name>.yaml`
3. Run `distillery distill recipes/<your-spirit-name>.yaml` locally
4. Verify it passes its own Tasting Notes (`distillery taste`)
5. Add an entry to `site/src/data/cellar.json` with your stats + R2 download URL
6. Open a PR with the Spirit's `.pt` file (or a link to where it's hosted)

Every Spirit ships with **honest** Tasting Notes including failure cases. We auto-generate them; we don't hide weaknesses.

## What's coming

- ✅ v0.1 — Tool-calling Spirits via Gemini (this release)
- 🚧 v0.2 — Claude teacher, GGUF export, byte-level BPE tokenizer
- 🚧 v0.3 — Classification Spirits, RAG-routing Spirits
- 🚧 v0.4 — Quantization-aware distillation
- 🚧 v0.5 — Browser-WASM bottling (run Spirits in the browser)
- 🚧 v1.0 — Hosted distillation as a service (free tier)

## Status

**v0.1** is a working alpha. The engine produces real, evaluable models — but argument-value accuracy (`exact_call_accuracy`) is still the weak spot pending byte-level tokenizer + more training data. Use it for prototypes; production work needs v0.2+.

## Development

```bash
# Clone
git clone https://github.com/the-distillery/distillarium
cd distillarium

# Install package editable
pip install -e ./distillery_pkg[gemini,dev]

# Run tests
cd distillery_pkg && pytest tests/ -p no:anchorpy

# Build site locally
cd ../site && npm install && npm run dev    # → http://localhost:4321

# Release a new package version
cd ../distillery_pkg
# bump pyproject.toml version, then:
./release.sh check        # build + twine check (no upload)
./release.sh test         # publish to TestPyPI
./release.sh publish      # publish to real PyPI (with confirmation prompt)
```

See [DEPLOY.md](DEPLOY.md) for the full Cloudflare Pages + R2 deployment playbook.

## License

MIT. See [LICENSE](LICENSE).

## Citing

If `distillarium` helps a paper or production system, a citation is appreciated but not required:

```bibtex
@software{distillarium2026,
  title  = {Distillarium: Distill any task into a pocket-sized Spirit},
  author = {The Distillery},
  year   = {2026},
  url    = {https://distillarium.app}
}
```

---

<p align="center">
  Built on top of an autonomous research-to-product pipeline that surfaces<br/>
  interesting work from arXiv, Hacker News, and GitHub daily.<br/>
  The first reference Spirit (<a href="https://distillarium.app/cellar/needle">Needle</a>)
  started life as a Show HN paper.
</p>
