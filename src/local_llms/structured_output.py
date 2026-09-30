"""Parse an entire JSON response and validate it without running generated code."""

import json
import re

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def parse_structured(text, schema):
    Draft202012Validator.check_schema(schema)
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        path = ".".join(map(str, exc.absolute_path)) or "<root>"
        raise ValueError(f"Response schema error at {path}: {exc.message}") from exc
    return value


def _reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")
