"""trainer evaluate — evaluate a trained model checkpoint."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Evaluate a trained model checkpoint on the test split.")
console = Console()


@app.callback(invoke_without_command=True)
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to the .pt checkpoint file."),
    dataset_dir: Path = typer.Option(
        Path("data/datasets"), "--dataset-dir", "-d", help="Dataset directory."
    ),
    output_dir: Path = typer.Option(
        Path("reports/evaluation"), "--output-dir", "-o", help="Where to write the report."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model architecture name."),
) -> None:
    """Run evaluation metrics on a checkpoint and generate a report."""
    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found:[/red] {checkpoint}")
        raise typer.Exit(1)

    console.print(f"[bold]Evaluating checkpoint:[/bold] {checkpoint}")
    console.print(
        "[yellow]Implement evaluate() call with a concrete BaseEvaluator subclass.[/yellow]"
    )
    console.print(f"Reports would be written to: {output_dir}")
