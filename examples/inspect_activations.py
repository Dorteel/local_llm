import torch

from local_llms.cli import load_from_args, model_parser


def main():
    parser = model_parser("Inspect native PyTorch modules, hidden states, and attention")
    parser.add_argument("--module", default="model.layers.0", help="Exact named_modules() path")
    parser.add_argument("--list-modules", action="store_true")
    parser.add_argument("--attentions", action="store_true")
    parser.add_argument("--prompt", default="The sky is blue.")
    args = parser.parse_args()
    model, tokenizer = load_from_args(args, attention="eager" if args.attentions else None)
    if args.list_modules:
        print(model)
        print("\n".join(dict(model.named_modules())))
        return
    modules = dict(model.named_modules())
    if args.module not in modules:
        parser.error("Unknown module; use --list-modules to inspect names")
    captured = {}

    def capture(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured[args.module] = tensor.detach().cpu()

    handle = modules[args.module].register_forward_hook(capture)
    try:
        inputs = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            result = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=args.attentions,
                use_cache=False,
            )
    finally:
        handle.remove()
    print(args.module, tuple(captured[args.module].shape))
    print("Hidden states:", [tuple(t.shape) for t in result.hidden_states])
    if args.attentions:
        print(
            "Attentions:",
            [tuple(t.shape) if t is not None else None for t in (result.attentions or [])],
        )


if __name__ == "__main__":
    main()
