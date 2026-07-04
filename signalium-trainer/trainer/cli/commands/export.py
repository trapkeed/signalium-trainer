"""trainer export — export a trained model to production formats."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Export a trained model to TorchScript and/or ONNX.")
console = Console()


@app.callback(invoke_without_command=True)
def export(
    checkpoint: Path = typer.Argument(..., help="Path to the .pt checkpoint file."),
    output_dir: Path = typer.Option(
        Path("exports"), "--output-dir", "-o", help="Export output directory."
    ),
    torchscript: bool = typer.Option(True, "--torchscript/--no-torchscript", help="Export TorchScript."),
    onnx: bool = typer.Option(True, "--onnx/--no-onnx", help="Export ONNX."),
    onnx_opset: int = typer.Option(17, "--onnx-opset", help="ONNX opset version."),
) -> None:
    """Export model artifacts for production deployment."""
    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found:[/red] {checkpoint}")
        raise typer.Exit(1)

    console.print(f"[bold]Exporting checkpoint:[/bold] {checkpoint}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"TorchScript: {torchscript} | ONNX: {onnx} (opset {onnx_opset})")
    console.print(
        "[yellow]Wire up TorchScriptExporter and ONNXExporter with your concrete model class.[/yellow]"
    )
