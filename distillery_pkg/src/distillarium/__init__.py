"""distillarium — Distill any task into a pocket-sized Spirit. Pure model. No API.

Public API:
    Recipe        — YAML-loaded distillation config
    distill()     — run a recipe end-to-end, return a Spirit
    taste()       — evaluate a Spirit on held-out cuts, return Tasting Notes
    bottle()      — export a Spirit (ONNX, GGUF, etc.)
    load_spirit() — load a saved Spirit from disk
    Spirit        — bottled model artifact (model + tokenizer + recipe + metrics)
"""

from distillarium.recipe import Recipe
from distillarium.spirit import Spirit, load_spirit
from distillarium.pipeline import distill, taste, bottle

__version__ = "0.1.0"

__all__ = [
    "Recipe",
    "Spirit",
    "load_spirit",
    "distill",
    "taste",
    "bottle",
    "__version__",
]
