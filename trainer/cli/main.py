"""Signalium Trainer CLI — entry point for all subcommands."""

from __future__ import annotations

import typer
from rich.console import Console

from trainer.cli import commands

app = typer.Typer(
    name="trainer",
    help="Signalium Trainer — deep learning training framework CLI.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

app.add_typer(commands.download_app, name="download")
app.add_typer(commands.validate_app, name="validate")
app.add_typer(commands.train_app, name="train")
app.add_typer(commands.evaluate_app, name="evaluate")
app.add_typer(commands.export_app, name="export")
app.add_typer(commands.upload_app, name="upload")
app.add_typer(commands.resume_app, name="resume")
app.add_typer(commands.benchmark_app, name="benchmark")
app.add_typer(commands.clean_app, name="clean")
app.add_typer(commands.doctor_app, name="doctor")


@app.command("version")
def version_cmd() -> None:
    """Print the trainer version."""
    from trainer import __version__

    console.print(f"signalium-trainer [bold green]{__version__}[/bold green]")


if __name__ == "__main__":
    app()
