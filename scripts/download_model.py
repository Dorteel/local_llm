"""The only entry point that downloads model files."""

import argparse

from local_llms.models import download_model, model_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--cache-dir")
    parser.add_argument("--info", action="store_true", help="Print selection without downloading")
    args = parser.parse_args()
    print(model_spec(args.model, args.config))
    if not args.info:
        print(download_model(args.model, config=args.config, cache_dir=args.cache_dir))


if __name__ == "__main__":
    main()
