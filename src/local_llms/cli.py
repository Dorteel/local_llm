"""Shared CLI options only; experimental behavior stays in each example."""

import argparse

from .models import load_model


def model_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("model")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--quantization", choices=["4bit", "8bit"])
    parser.add_argument("--adapter", help="Local PEFT adapter directory")
    return parser


def load_from_args(args, **kwargs):
    return load_model(
        args.model,
        config=args.config,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        adapter=args.adapter,
        **kwargs,
    )
