"""Execute the tutorial in a fresh Jupyter kernel using cached weights only.

Requires nbclient and ipykernel. Never changes the source notebook or downloads weights.
"""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", action="store_true", help="Also run one training step")
    parser.add_argument("--output", default="outputs/tutorial-checked.ipynb")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/local_llm_tutorial.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("DOWNLOAD_MODEL ="):
                lines[i] = "DOWNLOAD_MODEL = False  # Offline verification"
            if line.startswith("RUN_LORA ="):
                lines[i] = f"RUN_LORA = {args.lora!r}"
        cell.source = "\n".join(lines)
    nbformat.validate(notebook)
    # Inherited by the fresh kernel and the optional training subprocess.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    def report(cell_index, **kwargs):
        print(f"Executing cell {cell_index + 1}", flush=True)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="local-llms",
        resources={"metadata": {"path": str(root)}},
        on_cell_start=report,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Passed {count} code cells; executed notebook: {output}")


if __name__ == "__main__":
    main()
