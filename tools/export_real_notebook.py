"""Execute a small analysis notebook against an existing, sealed real run.

No downloads or training occur. Every displayed text output is captured from the
corresponding Python cell at export time, rather than copied from a report.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path


def export_notebook(dataset: Path, run: Path, output: Path) -> Path:
    project = Path(__file__).resolve().parents[1]
    dataset, run = dataset.resolve(), run.resolve()
    # The notebook stays portable when distributed with the repository.
    relative_dataset, relative_run = (
        str(dataset.relative_to(project)),
        str(run.relative_to(project)),
    )
    code = [
        "\n".join(
            [
                "from pathlib import Path",
                "import json",
                "import pandas as pd",
                "from f1ml.storage import verify_directory",
                "from f1ml_real.lab import RealDataset, load_run",
                "ROOT = Path.cwd()",
                "if not (ROOT / 'src').exists(): ROOT = ROOT.parent",
                f"DATASET = ROOT / {relative_dataset!r}",
                f"RUN = ROOT / {relative_run!r}",
                "source = RealDataset(DATASET)",
                "saved = load_run(RUN)",
                "source_receipt = verify_directory(DATASET)",
                "run_receipt = verify_directory(RUN)",
                "print('Source:', source_receipt['source'])",
                "print('Dataset ID:', source_receipt['id'])",
                "print('Run ID:', run_receipt['id'])",
                "print('Evaluation:', source_receipt['mode'])",
                "print(source_receipt['exposure'])",
            ]
        ),
        "\n".join(
            [
                "events = pd.read_parquet(DATASET / 'events.parquet')",
                "rosters = pd.read_parquet(DATASET / 'rosters.parquet')",
                "labels = pd.read_parquet(DATASET / 'labels.parquet')",
                "coverage = labels.groupby(['event_id', 'task']).agg(",
                "    entrants=('driver_id', 'size'),",
                "    known_numeric_ranks=('official_position', 'count'),",
                "    observed_outcomes=('label_observed', 'sum'))",
                "print(coverage.to_string())",
                "print('Excluded feature groups:', ', '.join(source_receipt['missing_groups']))",
                "print('Availability:', source_receipt['availability_semantics'])",
            ]
        ),
        "\n".join(
            [
                "config = json.loads((RUN / 'config.json').read_text())",
                "print(json.dumps(config, indent=2, sort_keys=True))",
                "assert (RUN / 'prediction_lock.json').exists()",
                "print('Configuration and forecasts were sealed before target scoring.')",
            ]
        ),
        "\n".join(
            [
                "development = pd.read_parquet(RUN / 'development_metrics.parquet')",
                "print('DEVELOPMENT: selection and calibration-fit stages')",
                "print(development.to_string(index=False))",
                "metrics = pd.read_parquet(RUN / 'metrics.parquet')",
                "print('SELECTED HISTORICAL TARGET')",
                "print(metrics.to_string(index=False))",
                "print('Development and selected historical-target metrics are separate rows.')",
                "print('Rank metrics use printed ranks; NC/DSQ tail ranks stay unknown.')",
            ]
        ),
        "\n".join(
            [
                "predictions = pd.read_parquet(RUN / 'predictions.parquet')",
                "print(predictions.to_string(index=False))",
                "print('Consulted historical outcomes: exploratory replay; no untouched holdout.')",
            ]
        ),
    ]
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Real F1 historical model test\n",
                "Executed analysis of a sealed run. "
                "The three learned families are linear/logistic, "
                "random forest, and histogram gradient boosting. "
                "Simple baselines remain comparison references.\n\n",
                "Training implementation: `src/f1ml_real/lab.py`. Reproduction and interpretation: "
                "`docs/real-model-lab.md`. "
                "Run this notebook from the repository root or `notebooks/`.\n\n",
                "No raw FIA PDFs are redistributed here; "
                "the dataset retains public document URLs and SHA-256 hashes.\n",
            ],
        }
    ]
    namespace = {"__name__": "__notebook__"}
    for index, source in enumerate(code, 1):
        captured = io.StringIO()
        with redirect_stdout(captured):
            exec(compile(source, f"real-f1-notebook-cell-{index}", "exec"), namespace)
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": index,
                "source": source.splitlines(keepends=True),
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": captured.getvalue().splitlines(keepends=True),
                    }
                ],
            }
        )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": __import__("platform").python_version()},
            "f1ml": {
                "dataset_id": namespace["source_receipt"]["id"],
                "run_id": namespace["run_receipt"]["id"],
                "executed_by": "tools/export_real_notebook.py",
            },
        },
    }
    # Cell IDs are required for portable nbformat 4.5 documents.
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"real-f1-{i:02d}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_notebook(args.dataset, args.run, args.output))


if __name__ == "__main__":
    main()
