"""One supervised optimizer step, adapter save/reload, disable, and detach."""

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from local_llms.cli import load_from_args, model_parser


def main():
    parser = model_parser(__doc__)
    parser.add_argument("--output", default="outputs/lora-smoke")
    args = parser.parse_args()
    if args.quantization or args.adapter:
        parser.error("This training smoke uses an unquantized base model without an adapter")
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new --output directory")
    if args.dtype == "auto":
        args.dtype = "float32"
    base, tokenizer = load_from_args(args)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        ),
    )
    model.print_trainable_parameters()
    prompt = [{"role": "user", "content": "What is two plus two?"}]
    prefix = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full = tokenizer.apply_chat_template(
        prompt + [{"role": "assistant", "content": "Two plus two is four."}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
    )
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Chat template does not support this simple prefix masking recipe")
    ids = torch.tensor([full], device=model.device)
    labels = ids.clone()
    labels[:, : len(prefix)] = -100
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    model.train()
    loss = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False
    ).loss
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print("One-step loss:", loss.item())
    model.save_pretrained(args.output)
    base = model.unload()  # Detach without merging learned weights into the base.
    restored = PeftModel.from_pretrained(base, args.output, local_files_only=True)
    restored.eval()
    with torch.inference_mode():
        print("Reloaded logits:", tuple(restored(input_ids=ids).logits.shape))
        with restored.disable_adapter():
            print("Base logits:", tuple(restored(input_ids=ids).logits.shape))
    restored.unload()
    print("Adapter saved, reloaded, temporarily disabled, and detached:", args.output)


if __name__ == "__main__":
    main()
