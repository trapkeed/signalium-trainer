"""trainer benchmark — benchmark DataLoader and model throughput."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Benchmark DataLoader and inference throughput.")
console = Console()


@app.callback(invoke_without_command=True)
def benchmark(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name to benchmark."),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size."),
    num_batches: int = typer.Option(50, "--num-batches", help="Number of batches to measure."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader worker count."),
) -> None:
    """Measure DataLoader throughput and model forward-pass latency."""
    import time

    import torch

    from trainer.utils.device import resolve_device
    from trainer.utils.gpu import get_gpu_info

    device = resolve_device("auto")
    gpu_info = get_gpu_info()

    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Device", str(device))
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if gpu_info.get("available"):
        for dev in gpu_info["devices"]:
            table.add_row(f"GPU {dev['index']}", f"{dev['name']} ({dev['total_memory_gb']} GB)")

    table.add_row("Batch Size", str(batch_size))
    table.add_row("Num Batches", str(num_batches))
    table.add_row("Num Workers", str(num_workers))

    if model:
        from trainer.registry import ModelRegistry

        try:
            model_cls = ModelRegistry.get(model)
            table.add_row("Model", model_cls.__name__)
        except KeyError:
            table.add_row("Model", f"[red]{model} (not found)[/red]")

    console.print(table)
    console.print("[yellow]Wire up real DataLoader and model for throughput measurement.[/yellow]")
