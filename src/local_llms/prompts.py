"""File-based string.Template prompts; JSON braces remain literal."""

from pathlib import Path
from string import Template


def render_prompt(path, **variables):
    template = Template(Path(path).read_text(encoding="utf-8"))
    try:
        return template.substitute(variables)
    except KeyError as exc:
        raise ValueError(f"Prompt {path}: missing variable {exc.args[0]!r}") from exc
    except ValueError as exc:
        raise ValueError(f"Prompt {path}: invalid template; escape literal dollars as $$") from exc
