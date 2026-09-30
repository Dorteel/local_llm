import json
from pathlib import Path

from local_llms.cli import load_from_args, model_parser
from local_llms.inference import generate
from local_llms.prompts import render_prompt
from local_llms.structured_output import parse_structured


def main():
    parser = model_parser("Generate and validate coding JSON; never execute it")
    parser.add_argument("--task", default="Write a function that returns the square of an integer.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    prompt = render_prompt("prompts/templates/structured_coding.txt", task=args.task)
    schema = json.loads(Path("prompts/examples/coding.schema.json").read_text())
    model, tokenizer = load_from_args(args)
    response = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
    try:
        result = parse_structured(response, schema)
    except ValueError as exc:
        parser.exit(1, f"{exc}\nRaw response:\n{response}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
