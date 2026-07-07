"""CLI command subapps — one module per subcommand group."""

from trainer.cli.commands.benchmark import app as benchmark_app
from trainer.cli.commands.clean import app as clean_app
from trainer.cli.commands.doctor import app as doctor_app
from trainer.cli.commands.download import app as download_app
from trainer.cli.commands.evaluate import app as evaluate_app
from trainer.cli.commands.export import app as export_app
from trainer.cli.commands.resume import app as resume_app
from trainer.cli.commands.train import app as train_app
from trainer.cli.commands.upload import app as upload_app
from trainer.cli.commands.validate import app as validate_app

__all__ = [
    "benchmark_app",
    "clean_app",
    "doctor_app",
    "download_app",
    "evaluate_app",
    "export_app",
    "resume_app",
    "train_app",
    "upload_app",
    "validate_app",
]
