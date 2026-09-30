"""Catalog, explicit downloads, and offline loading of ordinary HF models."""

from pathlib import Path

import yaml

# Use the identical file selection for downloads and offline completeness checks.
MODEL_FILE_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "*.jinja",
    "README.md",
    "LICENSE*",
]


def load_catalog(path="configs/models.yaml"):
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ValueError("Catalog must contain a 'models' mapping")
    if not data["models"]:
        raise ValueError("Catalog must contain at least one model")
    for name, spec in data["models"].items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise ValueError("Model names must be strings and entries must be mappings")
        for field in ("repo_id", "revision"):
            if not isinstance(spec.get(field), str) or not spec[field].strip():
                raise ValueError(f"Model {name}: '{field}' must be a nonempty string")
        purposes = spec.get("purpose")
        if (
            not isinstance(purposes, list)
            or not purposes
            or not all(isinstance(p, str) and p.strip() for p in purposes)
        ):
            raise ValueError(f"Model {name}: 'purpose' must be a nonempty list of strings")
    return data["models"]


def model_spec(name, config="configs/models.yaml"):
    catalog = load_catalog(config)
    if name not in catalog:
        raise ValueError(f"Unknown model {name!r}; choose from: {', '.join(catalog)}")
    return catalog[name]


def download_model(name, *, config="configs/models.yaml", cache_dir=None):
    from huggingface_hub import snapshot_download

    spec = model_spec(name, config)
    return snapshot_download(
        spec["repo_id"],
        revision=spec["revision"],
        cache_dir=cache_dir,
        allow_patterns=MODEL_FILE_PATTERNS,
    )


def load_model(
    name,
    *,
    config="configs/models.yaml",
    cache_dir=None,
    device="auto",
    dtype="auto",
    quantization=None,
    attention=None,
    adapter=None,
):
    """Return (model, tokenizer). Never download weights, tokenizers, or adapters."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    spec = model_spec(name, config)
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    if dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("Unsupported dtype")
    dtype = ("float32" if device == "cpu" else "float16") if dtype == "auto" else dtype
    if quantization not in (None, "4bit", "8bit"):
        raise ValueError("quantization must be 4bit or 8bit")
    if quantization and device != "cuda":
        raise ValueError("This project's quantization path requires CUDA")
    if adapter is not None and not Path(adapter).is_dir():
        raise ValueError("adapter must be an existing local directory")
    try:
        snapshot = snapshot_download(
            spec["repo_id"],
            revision=spec["revision"],
            cache_dir=cache_dir,
            local_files_only=True,
            allow_patterns=MODEL_FILE_PATTERNS,
        )
    except OSError as exc:
        raise ValueError(
            f"Required model files are missing or incomplete in the cache. "
            f"Run: python scripts/download_model.py {name} "
            f"--config {str(config)!r}"
            + (f" --cache-dir {str(cache_dir)!r}" if cache_dir is not None else "")
        ) from exc
    options = dict(
        local_files_only=True,
        trust_remote_code=False,
        dtype=getattr(torch, dtype),
        use_safetensors=True,
    )
    if attention:
        options["attn_implementation"] = attention
    if quantization:
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        )
        options["device_map"] = {"": device}
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(snapshot, **options)
    if not quantization:
        model.to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, local_files_only=True, is_trainable=False)
    model.eval()
    return model, tokenizer
