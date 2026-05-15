"""The pipeline — top-level distill / taste / bottle entrypoints.

These are the high-level functions a user calls. They orchestrate the engine,
teacher, tasting, and bottling modules.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from distillarium.engine.core import AttentionOnlyTransformer
from distillarium.engine.inference import FunctionCallGenerator
from distillarium.engine.tokenizer import FunctionCallTokenizer
from distillarium.engine.trainer import FunctionCallTrainer
from distillarium.recipe import Recipe
from distillarium.spirit import Spirit
from distillarium.tasting.evaluator import evaluate
from distillarium.teacher import get_teacher
from distillarium.teacher.base import load_distilled


def _build_corpus(data: list[tuple[str, list[dict], str]]) -> list[str]:
    """All text the tokenizer needs to learn vocabulary over."""
    out = []
    for utt, tools, target_json in data:
        out.append(f"[QUERY] {utt} [/QUERY]")
        out.append(f"[CALL] {target_json} [/CALL]")
        for t in tools:
            out.append(f"[TOOL] {t['name']} [/TOOL]")
    return out


def _split_cuts(
    data: list[tuple[str, list[dict], str]],
    train_frac: float,
    seed: int,
) -> tuple[list, list]:
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    n_train = int(len(data) * train_frac)
    train = [data[i] for i in idx[:n_train]]
    evald = [data[i] for i in idx[n_train:]]
    return train, evald


def distill(
    recipe: Recipe,
    out_dir: str | Path = "spirits/",
    mash_path: str | Path | None = None,
    verbose: bool = True,
) -> Spirit:
    """Run a recipe end-to-end. Returns a trained Spirit.

    Steps:
        1. Generate (or load) the Mash via the teacher
        2. Train the student tokenizer on the corpus
        3. Build the student model
        4. Run the still (training loop)
        5. Taste (held-out eval)
        6. Return a Spirit object (model + tokenizer + recipe + metrics)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mash — generate or load
    if mash_path is None:
        mash_path = out_dir / f"mash_{recipe.name.replace('.', '_')}.jsonl"
    mash_path = Path(mash_path)

    if not mash_path.exists():
        if verbose:
            print(f"⚗  Distilling {recipe.mash.total_examples} examples from "
                  f"{recipe.teacher.provider}:{recipe.teacher.model}...")
        teacher = get_teacher(
            recipe.teacher.provider,
            model=recipe.teacher.model,
            api_key=recipe.teacher.api_key,
            examples_per_call=recipe.mash.examples_per_call,
            tools_per_call_min=recipe.mash.tools_per_call.get("min", 3),
            tools_per_call_max=recipe.mash.tools_per_call.get("max", 6),
            seed=recipe.mash.seed,
        )
        stats = teacher.distill_to_file(
            mash_path,
            n_examples=recipe.mash.total_examples,
            examples_per_call=recipe.mash.examples_per_call,
        )
        if verbose:
            print(f"   Mash: {stats}")
    else:
        if verbose:
            print(f"⚗  Using existing Mash: {mash_path}")

    data = load_distilled(mash_path)
    if verbose:
        print(f"   Loaded {len(data)} (utt, tools, call) triples")

    # 2. Cuts
    train, evald = _split_cuts(data, recipe.cuts.train, recipe.mash.seed)
    if verbose:
        print(f"   Cuts: train={len(train)} eval={len(evald)}")

    # 3. Tokenizer
    if verbose:
        print(f"   Training tokenizer ({recipe.student.tokenizer})...")
    vocab_size = int(recipe.student.tokenizer.split("-")[-1]) \
        if "-" in recipe.student.tokenizer else 4096
    tokenizer = FunctionCallTokenizer(vocab_size=vocab_size)
    tokenizer.train(_build_corpus(data))

    # 4. Student model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"   Building student on {device}")
    model = AttentionOnlyTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=recipe.student.d_model,
        n_heads=recipe.student.n_heads,
        n_layers=recipe.student.n_layers,
        max_seq_len=recipe.student.max_seq_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"   Parameters: {n_params/1e6:.2f}M")

    # 5. The Still
    if verbose:
        print(f"🔥 Firing the still ({recipe.still.epochs} epochs, "
              f"batch={recipe.still.batch_size}, lr={recipe.still.lr})...")
    trainer = FunctionCallTrainer(
        model=model, tokenizer=tokenizer, lr=recipe.still.lr,
    )
    losses = trainer.train(
        train_data=train,
        epochs=recipe.still.epochs,
        batch_size=recipe.still.batch_size,
        max_len=recipe.student.max_seq_len,
    )
    if verbose:
        print(f"   Final loss: {losses[-1]:.4f}")
        print(f"   Loss curve: {[round(l, 3) for l in losses]}")

    # 6. Tasting
    if verbose:
        print(f"📝 Tasting on {min(recipe.tasting.held_out, len(evald))} held-out cuts...")
    generator = FunctionCallGenerator(model=model, tokenizer=tokenizer)
    metrics = evaluate(generator, evald, max_examples=recipe.tasting.held_out)
    if verbose:
        print(f"   Tool-name accuracy: {metrics['tool_name_accuracy']}")
        print(f"   Arg-key F1:         {metrics['arg_key_f1']}")
        print(f"   Exact-call:         {metrics['exact_call_accuracy']}")

    # 7. Bottle the Spirit
    spirit = Spirit(
        name=recipe.name,
        recipe=recipe,
        model=model,
        tokenizer=tokenizer,
        metrics=metrics,
        loss_curve=losses,
        n_params=n_params,
    )

    return spirit


def taste(
    spirit: Spirit,
    eval_data_path: str | Path,
    held_out: int = 100,
    teacher=None,
    teacher_held_out: int | None = None,
    previous: dict | str | Path | None = None,
) -> dict:
    """Run a fresh Tasting against held-out cuts.

    Args:
        spirit: the trained Spirit (student model + tokenizer).
        eval_data_path: JSONL of held-out (utt, tools, target_call) triples.
        held_out: cap on examples to score.
        teacher: optional teacher backend (string provider name or Teacher
            instance) to compute the teacher-baseline. If string, builds a
            Teacher from `spirit.recipe.teacher` config.
        teacher_held_out: optional smaller cap for the teacher pass; defaults
            to held_out. Each teacher call costs API tokens — set lower for
            cheap regression runs.
        previous: optional previous Tasting Notes (dict or path to a JSON
            file). Adds a `regression` block flagging metrics that got worse.

    Returns the new Tasting Notes dict and also writes it onto spirit.metrics.
    """
    data = load_distilled(eval_data_path)
    generator = FunctionCallGenerator(model=spirit.model, tokenizer=spirit.tokenizer)

    teacher_instance = None
    if teacher is not None:
        if isinstance(teacher, str):
            tcfg = spirit.recipe.teacher
            teacher_instance = get_teacher(
                teacher,
                model=tcfg.model,
                api_key=tcfg.api_key,
                examples_per_call=spirit.recipe.mash.examples_per_call,
                tools_per_call_min=spirit.recipe.mash.tools_per_call.get("min", 3),
                tools_per_call_max=spirit.recipe.mash.tools_per_call.get("max", 6),
                seed=spirit.recipe.mash.seed,
            )
        else:
            teacher_instance = teacher

    metrics = evaluate(
        generator,
        data,
        max_examples=held_out,
        teacher=teacher_instance,
        teacher_max_examples=teacher_held_out,
        previous=previous,
    )
    spirit.metrics = metrics
    return metrics


def bottle(spirit: Spirit, format: str = "pytorch", out: str | Path = None) -> Path:
    """Bottle a Spirit into a deployable format."""
    from distillarium.bottling import bottle_pytorch, bottle_onnx

    out = Path(out) if out else Path(f"spirits/{spirit.name}.{format}")
    fmt = format.lower()
    if fmt in ("pytorch", "pt"):
        return bottle_pytorch(spirit, out.with_suffix(".pt"))
    if fmt == "onnx":
        return bottle_onnx(spirit, out.with_suffix(".onnx"))
    raise ValueError(f"Unsupported bottling format: {format}. "
                     f"Supported in v0.1: pytorch, onnx")
