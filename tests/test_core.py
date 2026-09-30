import json
from pathlib import Path
from unittest.mock import patch

import pytest

from local_llms.models import download_model, load_catalog, model_spec
from local_llms.prompts import render_prompt
from local_llms.structured_output import parse_structured


def test_catalog():
    catalog = load_catalog()
    assert len(catalog) == 3
    assert "coding" in catalog["qwen2.5-coder-1.5b"]["purpose"]
    assert all(len(item["revision"]) == 40 for item in catalog.values())
    with pytest.raises(ValueError, match="Unknown model"):
        model_spec("missing")


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "models: {}",
        "models: []",
        "models: {x: 2}",
        "models: {x: {repo_id: x, revision: 12, purpose: [general]}}",
        "models: {x: {repo_id: x, revision: main, purpose: general}}",
    ],
)
def test_bad_catalog(tmp_path, content):
    path = tmp_path / "models.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_catalog(path)


def test_download_is_explicit_and_revisioned():
    with patch("huggingface_hub.snapshot_download", return_value="cached") as download:
        assert download_model("smollm2-135m") == "cached"
        assert download.call_args.kwargs["revision"] == model_spec("smollm2-135m")["revision"]
        assert "*.safetensors" in download.call_args.kwargs["allow_patterns"]


def test_prompts(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text('$task {"literal": true} $$5')
    assert render_prompt(path, task="Summarize") == 'Summarize {"literal": true} $5'
    with pytest.raises(ValueError, match="missing variable 'task'"):
        render_prompt(path)
    path.write_text("${broken")
    with pytest.raises(ValueError, match="invalid template"):
        render_prompt(path)


SCHEMA = json.loads(Path("prompts/examples/coding.schema.json").read_text())
VALID = dict(
    explanation="Squares",
    language="python",
    code="def square(x): return x*x",
    tests="assert square(2) == 4",
)


@pytest.mark.parametrize("fence", [False, True])
def test_structured_valid(fence):
    text = json.dumps(VALID)
    if fence:
        text = f"```json\n{text}\n```"
    assert parse_structured(text, SCHEMA) == VALID


@pytest.mark.parametrize("text", ["Here is {}", "{} trailing", "{", '{"x": NaN}'])
def test_bad_json(text):
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_structured(text, SCHEMA)


@pytest.mark.parametrize("changes", [{"language": "ruby"}, {"code": 42}, {"extra": True}])
def test_schema_errors(changes):
    with pytest.raises(ValueError, match="schema error"):
        parse_structured(json.dumps(VALID | changes), SCHEMA)


def test_missing_fields():
    with pytest.raises(ValueError, match="required property"):
        parse_structured("{}", SCHEMA)
