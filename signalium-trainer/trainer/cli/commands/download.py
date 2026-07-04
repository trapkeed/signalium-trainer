"""trainer download — download a dataset export from Signalium."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="Download a dataset export from Signalium.")
console = Console()


@app.callback(invoke_without_command=True)
def download(
    dataset_id: str = typer.Argument(..., help="Signalium dataset ID to download."),
    output_dir: Path = typer.Option(
        Path("data/datasets"), "--output-dir", "-o", help="Local directory to save the dataset."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a Hydra config override file."
    ),
    verify_checksums: bool = typer.Option(True, help="Verify SHA-256 checksums after download."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="SIGNALIUM_API_KEY", help="Signalium API key."
    ),
) -> None:
    """Download dataset files listed in the Signalium manifest."""
    from trainer.api.client import SignaliumClient
    from trainer.config.schema import APIConfig

    api_cfg = APIConfig(api_key=api_key)
    client = SignaliumClient(config=api_cfg, api_key=api_key)

    console.print(f"[bold]Fetching manifest for dataset:[/bold] {dataset_id}")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as p:
        task = p.add_task("Fetching manifest...", total=None)
        manifest = client.get_dataset_manifest(dataset_id)
        p.update(task, description=f"Found {len(manifest.files)} files")

    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest.files:
        if not manifest.download_url_base:
            console.print("[red]Manifest has no download_url_base — cannot download.[/red]")
            raise typer.Exit(1)
        url = f"{manifest.download_url_base.rstrip('/')}/{entry.path}"
        dest = output_dir / entry.path
        expected = entry.sha256 if verify_checksums else None
        client.download_dataset_file(url, dest, expected_sha256=expected)

    console.print(f"[green]Download complete.[/green] Files saved to {output_dir}")
