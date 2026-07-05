"""trainer train — start a training run."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Start a training run for a registered model architecture.")
console = Console()


@app.callback(invoke_without_command=True)
def train(
    model: str = typer.Argument(..., help="Registered model name (e.g. 'my_model')."),
    dataset_dir: Path = typer.Option(
        Path("data/datasets"), "--dataset-dir", "-d", help="Path to validated dataset directory."
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to YAML config overrides."
    ),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Override epoch count."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="MLflow run name."),
) -> None:
    """Launch a training run for the specified model architecture."""
    from trainer.config.schema import TrainerConfig
    from trainer.registry import ModelRegistry
    from trainer.utils.gpu import log_gpu_info

    console.print(f"[bold]Starting training run[/bold] for model: [cyan]{model}[/cyan]")

    cfg = TrainerConfig()
    cfg.training.seed = seed
    if epochs:
        cfg.training.epochs = epochs
    if run_name:
        cfg.mlflow.run_name = run_name

    log_gpu_info()

    try:
        model_cls = ModelRegistry.get(model)
    except KeyError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        f"[green]Model loaded:[/green] {model_cls.__name__} | "
        f"Config: environment={cfg.environment.value}"
    )
    console.print(
        "[yellow]Implement a concrete trainer (BaseTrainer subclass) and call engine.fit() here.[/yellow]"
    )
