# Engineering decisions

## Architecture: direct Hugging Face models
The original repository contained only an outdated llama.cpp/VLM README and no
implementation or submodule. Replace it with a task-independent Python package,
editable prompt assets, and explicit scripts. Return the actual model/tokenizer;
no backend hierarchy or task framework. GGUF/llama.cpp and serving engines are
useful future performance paths, but obscure the PyTorch internals needed here.

## Research and dependency selection (2026-09-30)
Public Hub metadata confirms the three catalog revisions and Apache-2.0 licenses.
Small established models are chosen for approachable memory requirements, not as
a claim of frontier quality: SmolLM2 135M for smoke runs, Qwen2.5 0.5B for general
chat, and Qwen2.5-Coder 1.5B for a capable small coding-specific baseline.

PyPI reports torch 2.14.0, transformers 5.17.0, accelerate 1.15.0, PEFT 0.21.1,
bitsandbytes 0.50.2, datasets 5.0.1 and TransformerLens 4.0.0. Use current bounded
major versions and a resolved lockfile. PyTorch provides transparent hooks;
Transformers handles architectures/chat templates; Hub handles revisioned caching;
Accelerate supports device placement; safetensors avoids pickle weights. YAML
and jsonschema cover human-editable configuration and output validation.
PEFT is optional instead of a custom LoRA implementation. datasets/TRL are deferred:
a tiny supervised batch needs neither, and real dataset pipelines should choose
formatting, masking, and evaluation deliberately. bitsandbytes is optional and
our initial tested policy limits quantized loading to CUDA; upstream supports
additional backends which can be added after hardware verification.

## Interpretability
Native PyTorch hooks and Transformers hidden states/attention outputs are the
minimal working path. Eager attention is used when collecting attention matrices;
short sequences matter because these tensors scale quadratically. TransformerLens
4.0 removed the old HookedTransformer.from_pretrained API and now advertises
TransformerBridge. Its additional dependency/API surface and model conversion
parity require separate validation. Defer its integration rather than claim
untested compatibility; native hooks need no extra interpretability dependency.

## Reproducibility and downloads
Catalog entries use immutable Hub commit hashes. Only the download script permits
network weight access. Inference resolves cached snapshots with local_files_only
and refuses remote model code. Downloads select safetensors/config/tokenizer assets,
not duplicate GGUF/ONNX/PyTorch weights. CPU defaults to float32, CUDA to float16;
explicit dtype overrides preserve a research path without quantization.

## Structured output and training scope
Strict JSON (optionally one complete Markdown fence) plus schema validation is
small and predictable. No brace extraction, retries, or constrained decoding;
invalid generations are expected, especially from the smallest model. Generated
code is never executed. LoRA uses a single supervised chat example with prompt
labels masked, one optimizer step, save/reload, temporary disable, and unload.
This verifies plumbing, not useful fine-tuning quality. Full SFT can use the same
raw model with all parameters enabled and a proper dataset/training loop.

## Sources
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
- https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes
- https://huggingface.co/docs/peft/main/en/developer_guides/lora
- https://github.com/TransformerLensOrg/TransformerLens
- https://pypi.org/ (release metadata queried for dependencies above)

## Verification and remaining limits
Validated on Python 3.12 with torch 2.14.0+cpu, Transformers 5.17.0, PEFT 0.21.1,
and Accelerate 1.15.0. The isolated CPU-wheel environment differs from the default
PyPI GPU-capable torch resolution in uv.lock; README distinguishes both install
paths. Host ROS PYTHONPATH injected launch-testing plugins requiring unrelated
packages; disable pytest plugin autoload instead of adding ROS dependencies.
Current Transformers chat tokenization returns a BatchEncoding by default, so the
LoRA recipe explicitly requests return_dict=False for prefix masking.

Offline tests exercise real tiny random Llama weights/tokenization, generation,
hooks, eager attention, adapter-only updates, save/reload numerical equivalence,
and inference/activation/LoRA script entry points. Catalog and prompt validation
are covered separately. CLI help, metadata-only download selection, Ruff, formatting,
lock consistency, and diff whitespace checks are also verified.
No pretrained weights were downloaded: downloads remain a user-initiated action.
Consequently pretrained output quality and actual catalog-model peak memory have
not been measured. CUDA/bitsandbytes and alternative Python versions remain
unverified. Follow the README's explicit SmolLM2 download smoke path next, then
measure memory/quality on the intended hardware before scaling experiments.

## Notebook tutorial
Added notebooks/local_llm_tutorial.ipynb as a guided consumer of the existing APIs,
with environment/kernel setup, model selection, prompt comparison, structured JSON,
native activation inspection, and the existing LoRA script. Explicit switches gate
downloads, training, and result-file writes. Jupyter remains a separately documented
interactive tool rather than an inference dependency. The notebook has no committed
outputs. Validated its nbformat schema and Python syntax, and executed all 13 default
code cells against a locally generated tiny model. Optional LoRA execution is covered
by the existing script tests, not this notebook execution; pretrained outputs remain
unverified. The synthetic model warned about its 128-token test context during the
long coding prompt; bundled models have larger context windows.

## Correction: filtered download versus offline snapshot completeness
The initial notebook check mocked snapshot_download and therefore missed a real
Hub cache failure. Downloads requested selected assets, while offline loading
requested the entire repository. Hub 1.33 correctly rejected the latter because
unrelated ONNX exports and .gitattributes were absent. Both operations now share
MODEL_FILE_PATTERNS; selected files still undergo Hub completeness validation.
Missing-cache errors also preserve custom config/cache arguments in the recovery
command. A regression builds a real local snapshot and cached tree listing,
confirms an unfiltered request fails, confirms filtered model inference succeeds,
and confirms removing selected safetensors weights still fails.

Reverified all 13 tutorial code cells in a fresh Jupyter kernel against the user's
already-cached SmolLM2 135M revision with HF_HUB_OFFLINE=1. Enabled the optional LoRA
branch: training, saving, reloading, disabling, and unloading all completed. Captured
layer shape was (1, 5, 576); LoRA trained 230,400 parameters with finite loss (~1.14).
The tiny pretrained model produced invalid coding JSON, which the tutorial handled
as an expected validation result, not an uncaught exception. The executed notebook
is outputs/tutorial-checked.ipynb. scripts/check_notebook.py makes this check
repeatable without changing the tutorial or fetching weights. All 27 tests and
Ruff checks passed. This supersedes the earlier synthetic-only notebook validation;
GPU/quantization and larger models remain unverified.
