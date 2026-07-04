"""trainer doctor — environment diagnostic check."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Run environment diagnostics and dependency checks.")
console = Console()


@app.callback(invoke_without_command=True)
def doctor() -> None:
    """Check the environment, GPU, dependencies, and API connectivity."""
    table = Table(title="Signalium Trainer — Environment Diagnostics")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Detail")

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 12)
    table.add_row(
        "Python ≥ 3.12",
        "[green]OK[/green]" if py_ok else "[red]FAIL[/red]",
        py_ver,
    )

    # PyTorch
    try:
        import torch
        table.add_row("PyTorch", "[green]OK[/green]", torch.__version__)
        cuda = torch.cuda.is_available()
        table.add_row(
            "CUDA",
            "[green]OK[/green]" if cuda else "[yellow]Unavailable[/yellow]",
            torch.version.cuda or "—",
        )
    except ImportError:
        table.add_row("PyTorch", "[red]MISSING[/red]", "pip install torch")
        cuda = False

    # MLflow
    try:
        import mlflow
        table.add_row("MLflow", "[green]OK[/green]", mlflow.__version__)
    except ImportError:
        table.add_row("MLflow", "[red]MISSING[/red]", "pip install mlflow")

    # ONNX
    try:
        import onnx
        table.add_row("ONNX", "[green]OK[/green]", onnx.__version__)
    except ImportError:
        table.add_row("ONNX", "[yellow]Missing[/yellow]", "pip install onnx")

    # Hydra
    try:
        import hydra
        table.add_row("Hydra", "[green]OK[/green]", hydra.__version__)
    except ImportError:
        table.add_row("Hydra", "[red]MISSING[/red]", "pip install hydra-core")

    # Loguru
    try:
        import loguru
        table.add_row("Loguru", "[green]OK[/green]", loguru.__version__)
    except ImportError:
        table.add_row("Loguru", "[red]MISSING[/red]", "pip install loguru")

    # GPU info
    if cuda:
        from trainer.utils.gpu import get_gpu_info
        info = get_gpu_info()
        for dev in info.get("devices", []):
            table.add_row(
                f"GPU {dev['index']}",
                "[green]OK[/green]",
                f"{dev['name']} | {dev['total_memory_gb']} GB",
            )

    # Env variables
    import os
    has_key = bool(os.getenv("SIGNALIUM_API_KEY") or os.getenv("SIGNALIUM_TOKEN"))
    table.add_row(
        "API credentials",
        "[green]Set[/green]" if has_key else "[yellow]Not set[/yellow]",
        "SIGNALIUM_API_KEY or SIGNALIUM_TOKEN",
    )

    console.print(table)
