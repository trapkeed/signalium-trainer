"""trainer clean — remove generated artefacts."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Remove generated artefacts (checkpoints, exports, logs, cache).")
console = Console()

_DIRS = {
    "checkpoints": Path("checkpoints"),
    "exports": Path("exports"),
    "logs": Path("logs"),
    "cache": Path("data/cache"),
    "reports": Path("reports"),
}


@app.callback(invoke_without_command=True)
def clean(
    all_: bool = typer.Option(False, "--all", "-a", help="Remove everything."),
    checkpoints: bool = typer.Option(False, "--checkpoints", help="Remove checkpoints."),
    exports: bool = typer.Option(False, "--exports", help="Remove exports."),
    logs: bool = typer.Option(False, "--logs", help="Remove logs."),
    cache: bool = typer.Option(False, "--cache", help="Remove dataset cache."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be removed."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Remove generated artefacts from the workspace."""
    import shutil

    targets: list[Path] = []
    if all_:
        targets = list(_DIRS.values())
    else:
        if checkpoints:
            targets.append(_DIRS["checkpoints"])
        if exports:
            targets.append(_DIRS["exports"])
        if logs:
            targets.append(_DIRS["logs"])
        if cache:
            targets.append(_DIRS["cache"])

    if not targets:
        console.print("[yellow]No targets selected. Use --all or specific flags.[/yellow]")
        raise typer.Exit(0)

    existing = [t for t in targets if t.exists()]
    if not existing:
        console.print("[green]Nothing to clean.[/green]")
        return

    console.print("[bold]Targets to remove:[/bold]")
    for t in existing:
        console.print(f"  {t}")

    if dry_run:
        console.print("[yellow]Dry run — nothing removed.[/yellow]")
        return

    if not yes:
        confirmed = typer.confirm("Proceed?", default=False)
        if not confirmed:
            console.print("Aborted.")
            raise typer.Exit(0)

    for t in existing:
        shutil.rmtree(t)
        console.print(f"[red]Removed:[/red] {t}")

    console.print("[green bold]Clean complete.[/green bold]")
