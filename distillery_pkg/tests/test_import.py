"""Smoke test: package imports + Recipe loads cleanly."""

from pathlib import Path


def test_top_level_imports():
    import distillarium
    from distillarium import Recipe, Spirit, distill, taste, bottle, load_spirit
    assert distillarium.__version__ == "0.1.2"


def test_engine_imports():
    from distillarium.engine import (
        AttentionOnlyTransformer,
        SchemaEncoder,
        FunctionCallTokenizer,
        FunctionCallTrainer,
        FunctionCallGenerator,
    )


def test_teacher_imports():
    from distillarium.teacher import Teacher, DistillExample, get_teacher
    # Concrete teacher requires the gemini extra; just check the type lookup
    try:
        from distillarium.teacher import GeminiTeacher
    except RuntimeError:
        pass  # google-genai not installed — that's fine for import-only test


def test_tasting_imports():
    from distillarium.tasting import evaluate, parse_generated_call


def test_bottling_imports():
    from distillarium.bottling import bottle_pytorch, bottle_onnx


def test_recipe_load_from_yaml():
    from distillarium import Recipe

    recipe_path = Path(__file__).parent.parent / "recipes" / "needle.tool-calling-v1.yaml"
    recipe = Recipe.from_file(recipe_path)
    assert recipe.name == "needle.tool-calling"
    assert recipe.version == 1
    assert recipe.teacher.provider == "gemini"
    assert recipe.mash.total_examples == 1000
    assert recipe.student.d_model == 384
    assert recipe.student.n_layers == 8


def test_recipe_defaults():
    from distillarium import Recipe

    r = Recipe.from_dict({"name": "test.minimal"})
    assert r.name == "test.minimal"
    assert r.teacher.provider == "gemini"
    assert r.student.d_model == 384
    assert r.still.epochs == 8


def test_recipe_round_trip():
    from distillarium import Recipe

    r = Recipe.from_dict({"name": "test.rt", "version": 2})
    d = r.to_dict()
    r2 = Recipe.from_dict(d)
    assert r2.name == r.name
    assert r2.version == r.version
    assert r2.student.d_model == r.student.d_model


def test_parse_generated_call_normalizes_whitespace():
    from distillarium.tasting import parse_generated_call

    noisy = [{" name ": " send_message ", " args ": {" contact ": " alice "}}]
    parsed = parse_generated_call(noisy)
    assert parsed is not None
    assert parsed["name"] == "send_message"
    assert parsed["args"]["contact"] == "alice"


def test_parse_generated_call_returns_none_for_empty():
    from distillarium.tasting import parse_generated_call
    assert parse_generated_call([]) is None
    assert parse_generated_call([None]) is None
    assert parse_generated_call([{"args": {}}]) is None  # no name
