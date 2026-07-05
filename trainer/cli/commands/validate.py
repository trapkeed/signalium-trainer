"""trainer validate — validate a downloaded dataset."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Validate a downloaded dataset (checksums, schema, splits).")
console = Console()


@app.callback(invoke_without_command=True)
def validate(
    dataset_dir: Path = typer.Argument(..., help="Path to the dataset directory."),
    no_checksums: bool = typer.Option(False, "--no-checksums", help="Skip checksum verification."),
    strict: bool = typer.Option(True, help="Exit with error code on validation failure."),
) -> None:
    """Validate a downloaded dataset directory."""
    from trainer.datasets.validation import DatasetValidator

    validator = DatasetValidator(
        dataset_dir=dataset_dir,
        verify_checksums=not no_checksums,
        fail_on_error=strict,
    )
    try:
        result = validator.validate()
        if result.passed:
            console.print("[green bold]Validation PASSED[/green bold]")
        else:
            console.print("[yellow bold]Validation completed with warnings[/yellow bold]")
    except ValueError as exc:
        console.print(f"[red bold]Validation FAILED:[/red bold] {exc}")
        raise typer.Exit(1)
