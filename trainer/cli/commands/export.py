"""trainer export — export a trained model checkpoint via the unified
S21A QDE export pipeline (weights, TorchScript, ONNX, config, calibration,
metadata, qde_manifest.json — all in one call, with an ArtifactRegistry
record for every file).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Export a trained model checkpoint to production artifacts.")
console = Console()


@app.callback(invoke_without_command=True)
def export(
    checkpoint: Path = typer.Argument(..., help="Path to the .pt checkpoint file."),
    model_version: str = typer.Option(
        ..., "--model-version", "-v", help="Version tag, e.g. 'v3.1'."
    ),
    output_dir: Path = typer.Option(
        Path("exports"), "--output-dir", "-o", help="Export output directory."
    ),
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        "-m",
        help=(
            "ModelRegistry key to instantiate before loading the checkpoint "
            "(required for TorchScript/ONNX export; without it only "
            "weights + config/calibration/metadata/manifest are exported)."
        ),
    ),
    export_formats: str = typer.Option(
        "torchscript,onnx", "--export-formats", help="Comma-separated: torchscript,onnx."
    ),
    dataset_version: str = typer.Option("", "--dataset-version"),
    feature_version: str = typer.Option("", "--feature-version"),
    certification_status: str = typer.Option("PENDING", "--certification-status"),
    promotion_status: str = typer.Option("NOT_DEPLOYED", "--promotion-status"),
) -> None:
    """Export a trained checkpoint's full artifact set for production deployment."""
    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found:[/red] {checkpoint}")
        raise typer.Exit(1)

    try:
        import torch
    except ImportError:
        console.print("[red]torch is required to export a checkpoint.[/red]")
        raise typer.Exit(1) from None

    import os

    from trainer.config.schema import TrainerConfig
    from trainer.exporters.pipeline import QDEExportPipeline
    from trainer.research_lab.artifact_registry import ArtifactRegistry
    from trainer.research_lab.rl_database import ResearchLabDatabase
    from trainer.traceability.context import _git_commit

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config") or TrainerConfig()

    model = None
    example_input_shape = None
    formats = tuple(f.strip() for f in export_formats.split(",") if f.strip())

    if model_name:
        from trainer.registry.registry import ModelRegistry

        model_cls = ModelRegistry.get(model_name)
        model_config = state.get("model_config")
        model = model_cls(config, model_config) if model_config is not None else model_cls(config)
        model.load_state_dict(state["model_state_dict"])
        example_input_shape = state.get("example_input_shape")
        if example_input_shape is None:
            console.print(
                "[yellow]No example_input_shape in checkpoint; "
                "skipping TorchScript/ONNX export.[/yellow]"
            )
            formats = ()
    else:
        console.print(
            "[yellow]No --model-name given; exporting weights + config/calibration/"
            "metadata/manifest only (no TorchScript/ONNX).[/yellow]"
        )
        formats = ()

    db_path = os.environ.get("RESEARCH_LAB_DB_PATH", "data/research_lab.db")
    registry = ArtifactRegistry(ResearchLabDatabase(db_path))
    pipeline = QDEExportPipeline(artifact_registry=registry)
    weights_path = None
    if model is None:
        weights_path = Path(output_dir) / model_version / "weights.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state["model_state_dict"], str(weights_path))

    result = pipeline.export_version(
        model_version=model_version,
        output_dir=output_dir,
        model=model,
        weights_path=weights_path,
        config=config,
        example_input_shape=example_input_shape,
        export_formats=formats,
        dataset_version=dataset_version,
        feature_version=feature_version,
        training_commit=_git_commit(short=True),
        certification_status=certification_status,
        promotion_status=promotion_status,
    )

    console.print(
        f"[bold green]Exported[/bold green] {len(result['files'])} artifact(s) to {result['output_dir']}"
    )
    console.print(
        f"Manifest: {result['manifest_path']} (sha256={result['manifest_sha256'][:12]}...)"
    )
    console.print(f"Registered artifacts: {result['artifacts']}")
