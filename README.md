# Local open-weight LLM experiments

A small Python repository for local inference, editable prompts, structured
responses, LoRA experiments, and inspecting model internals. Coding is one example
application. Models remain ordinary Hugging Face/PyTorch objects.

## Install

Use Python 3.10+ and run commands from the repository root. No model weights are
downloaded during installation or inference.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU installation; install your CUDA-compatible PyTorch wheel instead for GPU use.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

Choose the appropriate wheel using [PyTorch's installer](https://pytorch.org/get-started/locally/).
An old CUDA toolkit version from the original README is not a hardware requirement;
wheel support and the installed NVIDIA driver determine compatibility.

Core dependencies: PyTorch for tensors/hooks, Transformers for models and chat
formatting, Hub for revisioned downloads, Accelerate for placement/quantization,
safetensors for weights, PyYAML for configuration, and jsonschema for validation.
Optional installs:

```bash
python -m pip install -e '.[fine-tuning]'  # PEFT
python -m pip install -e '.[quantization]' # bitsandbytes; this project's path uses CUDA
```

`uv.lock` records exact dependency resolution. With [uv](https://docs.astral.sh/uv/),
`uv sync --locked --extra dev --extra fine-tuning` reproduces the default PyPI
resolution (which may include large CUDA packages on Linux). The CPU-wheel pip
recipe above is a smaller hardware-specific alternative, not an exact lock replay.
Record `python -m pip freeze`, the catalog revision, device, dtype, prompt, and
outputs with research results. Greedy decoding does not guarantee identical
numerics across hardware or library versions.

## Notebook tutorial

Open [the step-by-step notebook](notebooks/local_llm_tutorial.ipynb) for an interactive
walkthrough of inference, prompt comparisons, JSON validation, activation inspection,
and optional LoRA training. Its first section explains Jupyter installation and
kernel selection. Downloads and training require explicit opt-in.

After source updates, restart the notebook kernel before running all cells. To
verify the tutorial against an already downloaded SmolLM2 model in a fresh kernel:

```bash
python -m pip install nbclient ipykernel
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name local-llms --display-name "Local LLMs"
python scripts/check_notebook.py --lora
```

This check disables network model access, enables the optional LoRA section, and
writes an executed copy to `outputs/tutorial-checked.ipynb`. Omit `--lora` to check
only the inference/inspection path. It requires the notebook's selected model to
be cached and the appropriate dependencies installed.

## Models and explicit downloads

| Catalog key | Model | Weight storage approximately | Short-context working memory estimate |
| --- | --- | --- | --- |
| `smollm2-135m` | [SmolLM2 135M Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 0.27 GB at 16-bit; repository may store FP32 (~0.54 GB) | 1–2 GB CPU RAM; smallest smoke path |
| `qwen2.5-0.5b` | [Qwen2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 1 GB at 16-bit | 2–4 GB RAM or VRAM |
| `qwen2.5-coder-1.5b` | [Qwen2.5 Coder 1.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 3.1 GB at 16-bit | 4–6 GB VRAM; 8+ GB CPU RAM at FP32 |

All three selected revisions use Apache-2.0; consult linked model cards and licenses
for terms and limitations. Estimates are planning guidance, not measured peaks.
CPU float32 weights take twice the memory of float16 weights. Loading, KV cache,
long prompts, training gradients, and retained activations add memory. Small
models are convenient baselines and can give incorrect or malformed answers.

```bash
python scripts/download_model.py smollm2-135m --info # metadata only
python scripts/download_model.py smollm2-135m        # explicit network download
python examples/basic_inference.py smollm2-135m --device cpu --max-new-tokens 32
```

Downloads use the standard Hugging Face cache and immutable revisions from
`configs/models.yaml`; repeated downloads reuse it. Set `HF_HOME` or pass the same
`--cache-dir /path/to/cache` to download and example commands to change location.
Incomplete caches fail locally; rerun the download command to complete them.
Public examples need no login. For a gated model, accept its license on the Hub,
then run `hf auth login` (or set `HF_TOKEN` privately). Never commit tokens.
Remote Python model code is disabled and safetensors weights are required.

## Prompts and structured output

```bash
python examples/basic_inference.py smollm2-135m --prompt 'Give three ways to save water.'
python examples/basic_inference.py smollm2-135m \
  --template prompts/templates/summarize.txt --variables prompts/examples/summary.json
python examples/basic_inference.py smollm2-135m \
  --template prompts/templates/explain.txt --variables prompts/examples/explanation.json
python scripts/download_model.py qwen2.5-coder-1.5b
python examples/structured_coding.py qwen2.5-coder-1.5b \
  --task 'Write a function that tests whether a string is a palindrome.'
```

Structured coding validates `explanation`, `language`, `code`, and `tests` against
`prompts/examples/coding.schema.json`. Invalid JSON/schema results exit nonzero
with the error and raw response. Validation does not ensure correct code and never
executes it. Generation is not grammar-constrained; try a better model, clearer
prompt, or larger token budget if it fails.

Add a UTF-8 file under `prompts/templates/` using `$variable` or `${variable}`.
Literal JSON braces need no escaping; literal dollar signs use `$$`. Supply a JSON
object through `--variables`. Missing variables produce explicit errors.

## Hardware and direct Python access

Examples accept `--device auto|cpu|cuda`, `--dtype auto|float32|float16|bfloat16`,
`--quantization 4bit|8bit`, `--config`, `--cache-dir`, and `--adapter`.
Auto uses CUDA when available, otherwise CPU; default dtype is float16 on CUDA
and float32 on CPU. Choose bfloat16 only on supported hardware. Quantization is
optional, requires the extra above and CUDA in this initial implementation, and
changes the tensors inspected by interpretability experiments. It reduces weight
memory, not all activation/cache memory. No multi-GPU/offload policy is provided.

```python
from local_llms.models import load_model
from local_llms.inference import generate

model, tokenizer = load_model("smollm2-135m", device="cpu")
print(generate(model, tokenizer, "Explain gravity briefly."))
print(model.config)
# model.named_modules(), model.parameters(), tokenizer, and model(...) are accessible.
```

## LoRA and supervised fine-tuning

```bash
python -m pip install -e '.[fine-tuning]'
python examples/lora_example.py smollm2-135m --device cpu
python examples/basic_inference.py smollm2-135m --adapter outputs/lora-smoke
```

The example creates rank-4 adapters on `q_proj`/`v_proj`, masks the user prompt from
supervised labels, runs one optimizer step on a tiny factual answer, saves only the
adapter, reloads it, temporarily disables it, and detaches it without merging.
Use a fresh `--output` path for repeat runs. It uses float32 by default for a simple
training smoke test and rejects quantized/already-adapted inputs. Target module
names and chat prefix masking must be checked when adding architectures.

One step is a pipeline check, not a useful trained adapter. To extend it, add a
licensed dataset, train/validation split, batching with padding labels set to -100,
optimizer/scheduler and checkpoints, and independent evaluation. For full SFT,
use the base model directly with trainable parameters rather than wrapping it in
PEFT. For QLoRA, add `prepare_model_for_kbit_training` and verify backend support;
the current training example deliberately exercises unquantized LoRA.
Always load an adapter against the same base model and revision used to train it.
Use PEFT's `load_adapter`/`set_adapter` for multiple adapters and
`with model.disable_adapter():` for temporary base-model comparisons.

## Mechanistic interpretability

No extra dependency is needed for native hooks:

```bash
python examples/inspect_activations.py smollm2-135m --device cpu
python examples/inspect_activations.py smollm2-135m --attentions
python examples/inspect_activations.py smollm2-135m --list-modules
python examples/inspect_activations.py smollm2-135m --module model.layers.0.self_attn
```

The example captures a module output, prints its name and shape, and reports hidden
states. `--attentions` selects eager attention to expose matrices where supported.
Inputs are capped at 128 tokens to contain activation memory. Hooks are removed in
`finally`; adapt the capture function for modules returning non-tensor structures.
Prefer unquantized float32 when numerical fidelity matters. TransformerLens 4.0
has a new bridge API; integration/parity for these models is deferred, with reasoning
and upstream references in [notes.md](notes.md).

## Extend and test

Add a catalog entry with `repo_id`, immutable `revision`, and a nonempty `purpose`
list. Record its license and memory estimate. Native Transformers causal-LM
support, safetensors weights, and a usable chat template are required. Check asset
patterns in `download_model` for unusual tokenizers. No Python registry changes
are needed for ordinary supported models.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
ruff check .
```

The environment variable prevents unrelated host plugins (for example ROS launch
testing) from being auto-loaded. Tests never download weights. They cover catalog errors, template rendering, JSON
validation, explicit download configuration, offline loading/generation, real
activation/attention tensors, and a LoRA save/reload cycle using a tiny random
model created locally. PEFT tests skip if its optional extra is absent.
The three-command SmolLM2 download/inference sequence above is the real pretrained
model smoke path; follow it with the activation and LoRA examples for a fuller run.

Repository layout:

- `src/local_llms/`: catalog/loading, inference, prompts, JSON validation, shared CLI options.
- `configs/`: pinned model catalog.
- `scripts/`: explicitly initiated model downloads.
- `prompts/`: editable templates, variables, and schemas.
- `examples/`: inference, structured coding, adapter training, and activation capture.
- `tests/`: offline unit and tiny-model integration tests.
- `notes.md`: research sources, decisions, trade-offs, and verification limits.
