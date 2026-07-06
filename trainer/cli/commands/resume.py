"""trainer resume — resume training from a checkpoint."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Resume a training run from a saved checkpoint.")
console = Console()


@app.callback(invoke_without_command=True)
def resume(
    checkpoint: Path = typer.Argument(..., help="Path to the checkpoint to resume from."),
    model: str = typer.Option(..., "--model", "-m", help="Registered model name."),
    dataset_dir: Path = typer.Option(
        Path("data/datasets"), "--dataset-dir", "-d", help="Dataset directory."
    ),
    epochs: int | None = typer.Option(None, "--epochs", "-e", help="Total epochs to run."),
) -> None:
    """Resume training from a saved checkpoint file."""
    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found:[/red] {checkpoint}")
        raise typer.Exit(1)

    console.print(f"[bold]Resuming training from:[/bold] {checkpoint}")
    console.print(f"Model: {model} | Dataset: {dataset_dir}")
    console.print("[yellow]Pass checkpoint to engine.fit(resume_checkpoint=...) here.[/yellow]")
