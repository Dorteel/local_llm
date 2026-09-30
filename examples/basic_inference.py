import json
from pathlib import Path

from local_llms.cli import load_from_args, model_parser
from local_llms.inference import generate
from local_llms.prompts import render_prompt


def main():
    parser = model_parser("Run a chat prompt or an editable prompt template")
    parser.add_argument("--prompt", default="Explain why the sky appears blue in two sentences.")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--template")
    parser.add_argument("--variables", help="Path to a JSON object of template variables")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.variables and not args.template:
        parser.error("--variables requires --template")
    prompt = args.prompt
    if args.template:
        variables = json.loads(Path(args.variables).read_text()) if args.variables else {}
        prompt = render_prompt(args.template, **variables)
    model, tokenizer = load_from_args(args)
    print(
        generate(model, tokenizer, prompt, system=args.system, max_new_tokens=args.max_new_tokens)
    )


if __name__ == "__main__":
    main()
