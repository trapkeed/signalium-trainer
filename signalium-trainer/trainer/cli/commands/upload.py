"""trainer upload — upload model artifacts to Signalium."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Upload trained model artifacts to Signalium.")
console = Console()


@app.callback(invoke_without_command=True)
def upload(
    export_dir: Path = typer.Argument(..., help="Directory containing exported model files."),
    model_id: str = typer.Option(..., "--model-id", help="Signalium model ID."),
    version: str = typer.Option(..., "--version", help="Semantic version (e.g. 1.0.0)."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="MLflow run ID."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="SIGNALIUM_API_KEY", help="Signalium API key."
    ),
) -> None:
    """Upload model files, manifest, metrics, and logs to Signalium."""
    if not export_dir.exists():
        console.print(f"[red]Export directory not found:[/red] {export_dir}")
        raise typer.Exit(1)

    from trainer.api.client import SignaliumClient
    from trainer.config.schema import APIConfig

    api_cfg = APIConfig(api_key=api_key)
    client = SignaliumClient(config=api_cfg, api_key=api_key)

    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]manifest.json not found in {export_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Uploading model[/bold] {model_id} v{version}")
    console.print("[yellow]Implement file upload loop using client.upload_model_file().[/yellow]")
    console.print("[green]Upload complete (stub — wire up real upload loop).[/green]")
