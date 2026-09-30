"""Real tiny random PyTorch model, tokenizer, and PEFT tests; no Hub downloads."""

from unittest.mock import patch

import pytest

pytest.importorskip("torch")
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from local_llms.inference import generate
from local_llms.models import MODEL_FILE_PATTERNS, load_model


@pytest.fixture
def tiny_snapshot(tmp_path):
    raw = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<eos>": 1, "hello": 2, "user": 3, "assistant": 4, "system": 5},
            unk_token="<unk>",
        )
    )
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + "
        "message['content'] + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    tokenizer.save_pretrained(tmp_path)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=6,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            eos_token_id=1,
            pad_token_id=1,
        )
    )
    model.save_pretrained(tmp_path)
    return tmp_path


def test_offline_load_generate_and_hooks(tiny_snapshot):
    with patch("huggingface_hub.snapshot_download", return_value=str(tiny_snapshot)) as download:
        model, tokenizer = load_model("smollm2-135m", device="cpu", attention="eager")
    assert download.call_args.kwargs["local_files_only"] is True
    assert download.call_args.kwargs["allow_patterns"] == MODEL_FILE_PATTERNS
    assert next(model.parameters()).dtype == torch.float32
    assert isinstance(generate(model, tokenizer, "hello", max_new_tokens=2), str)
    captured = []
    handle = model.model.layers[0].register_forward_hook(
        lambda module, inputs, output: captured.append(output.shape)
    )
    try:
        with torch.inference_mode():
            output = model(
                input_ids=torch.tensor([[2, 2]]), output_hidden_states=True, output_attentions=True
            )
    finally:
        handle.remove()
    assert captured[0] == (1, 2, 16)
    assert len(output.hidden_states) == 2
    assert output.attentions[0].shape == (1, 2, 2, 2)


def test_missing_cache():
    with patch("huggingface_hub.snapshot_download", side_effect=OSError("missing")):
        with pytest.raises(ValueError, match="download_model.py"):
            load_model("smollm2-135m", device="cpu")


def test_cpu_quantization_rejected():
    with pytest.raises(ValueError, match="requires CUDA"):
        load_model("smollm2-135m", device="cpu", quantization="4bit")


def test_lora_step_save_reload_and_detach(tiny_snapshot, tmp_path):
    peft = pytest.importorskip("peft")
    base = LlamaForCausalLM.from_pretrained(tiny_snapshot)
    model = peft.get_peft_model(
        base, peft.LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj", "v_proj"])
    )
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    ids = torch.tensor([[2, 3, 2, 1]])
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert any(
        not torch.equal(params[n], p) for n, p in model.named_parameters() if p.requires_grad
    )
    assert all(
        torch.equal(params[n], p) for n, p in model.named_parameters() if not p.requires_grad
    )
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    adapter = tmp_path / "adapter"
    model.save_pretrained(adapter)
    base = model.unload()
    restored = peft.PeftModel.from_pretrained(base, adapter, local_files_only=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(input_ids=ids).logits, expected)
        with restored.disable_adapter():
            assert restored(input_ids=ids).logits.shape == expected.shape
    assert isinstance(restored.unload(), LlamaForCausalLM)


@pytest.mark.parametrize(
    "script,extra",
    [
        ("basic_inference", ["--max-new-tokens", "2"]),
        ("inspect_activations", ["--attentions"]),
        ("lora_example", []),
    ],
)
def test_example_entrypoints(tiny_snapshot, tmp_path, script, extra):
    import runpy
    import sys

    if script == "lora_example":
        pytest.importorskip("peft")
        extra = ["--output", str(tmp_path / "saved-adapter")]
    argv = [f"examples/{script}.py", "smollm2-135m", "--device", "cpu", *extra]
    with (
        patch.object(sys, "argv", argv),
        patch("huggingface_hub.snapshot_download", return_value=str(tiny_snapshot)),
    ):
        runpy.run_path(argv[0], run_name="__main__")


def test_filtered_snapshot_loads_without_unrelated_exports(tiny_snapshot, tmp_path):
    """Use a real Hub cache/tree listing, not a mocked snapshot_download result."""
    import json
    import shutil

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import IncompleteSnapshotError

    from local_llms.models import model_spec

    spec = model_spec("smollm2-135m")
    cache = tmp_path / "hub-cache"
    repo = cache / ("models--" + spec["repo_id"].replace("/", "--"))
    snapshot = repo / "snapshots" / spec["revision"]
    snapshot.mkdir(parents=True)
    files = {}
    for file in tiny_snapshot.iterdir():
        if file.is_file():
            shutil.copyfile(file, snapshot / file.name)
            files[file.name] = {"size": file.stat().st_size, "blob_id": "0" * 40}
    # These repository files were deliberately excluded from the download.
    files["onnx/model.onnx"] = {"size": 100, "blob_id": "1" * 40}
    files[".gitattributes"] = {"size": 100, "blob_id": "2" * 40}
    (repo / "trees").mkdir()
    (repo / "trees" / f"{spec['revision']}.json").write_text(
        json.dumps({"format_version": 1, "files": files})
    )
    with pytest.raises(IncompleteSnapshotError):
        snapshot_download(
            spec["repo_id"], revision=spec["revision"], cache_dir=cache, local_files_only=True
        )
    model, tokenizer = load_model("smollm2-135m", cache_dir=cache, device="cpu")
    assert isinstance(generate(model, tokenizer, "hello", max_new_tokens=2), str)
    # Filtering must still catch actually missing selected weights.
    (snapshot / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing or incomplete") as error:
        load_model("smollm2-135m", cache_dir=cache, device="cpu")
    assert str(cache) in str(error.value)
