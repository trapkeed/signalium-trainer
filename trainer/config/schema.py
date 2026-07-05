"""Structured configuration schemas using Hydra / dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    RUNPOD = "runpod"


class OptimizerType(str, Enum):
    """Supported optimizers."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADAMAX = "adamax"


class SchedulerType(str, Enum):
    """Supported LR schedulers."""

    COSINE = "cosine"
    STEP = "step"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    EXPONENTIAL = "exponential"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    ONE_CYCLE = "one_cycle"
    LINEAR = "linear"
    NONE = "none"


@dataclass
class APIConfig:
    """Signalium REST API connection settings."""

    base_url: str = "https://api.signalium.io"
    timeout: int = 60
    max_retries: int = 5
    retry_backoff_factor: float = 2.0
    api_key: Optional[str] = None
    token: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset location and versioning."""

    dataset_dir: str = "data/datasets"
    output_dir: str = "data/outputs"
    cache_dir: str = "data/cache"
    dataset_version: Optional[str] = None
    validate_checksums: bool = True
    fail_on_validation_error: bool = True


@dataclass
class DataLoaderConfig:
    """PyTorch DataLoader settings."""

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = False
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class OptimizerConfig:
    """Optimizer hyper-parameters."""

    type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    eps: float = 1e-8
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    """LR scheduler settings."""

    type: SchedulerType = SchedulerType.COSINE
    warmup_epochs: int = 0
    T_max: int = 100
    eta_min: float = 1e-6
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5
    factor: float = 0.5
    T_0: int = 10
    T_mult: int = 1
    max_lr: float = 1e-2
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    total_steps: Optional[int] = None


@dataclass
class CheckpointConfig:
    """Checkpoint persistence settings."""

    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    keep_top_k: int = 3
    save_best: bool = True
    save_latest: bool = True
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_rng_state: bool = True
    save_training_config: bool = True


@dataclass
class ReproducibilityConfig:
    """Reproducibility tracking settings."""

    enabled: bool = True
    fingerprint_dir: str = "checkpoints/fingerprints"
    dataset_version: Optional[str] = None
    feature_version: Optional[str] = None
    label_version: Optional[str] = None
    verify_on_resume: bool = True


@dataclass
class TrainingConfig:
    """Core training hyper-parameters."""

    epochs: int = 100
    seed: int = 42
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1


@dataclass
class GPUConfig:
    """GPU / device settings."""

    device: str = "auto"
    num_gpus: int = 1
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


@dataclass
class MLflowConfig:
    """MLflow experiment tracking."""

    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str = "signalium-trainer"
    run_name: Optional[str] = None
    log_artifacts: bool = True
    log_models: bool = True
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Structured logging settings."""

    level: str = "INFO"
    format: str = "json"
    log_dir: str = "logs"
    rotation: str = "100 MB"
    retention: str = "7 days"
    backtrace: bool = True
    diagnose: bool = False


@dataclass
class ExportConfig:
    """Model export settings."""

    output_dir: str = "exports"
    export_torchscript: bool = True
    export_onnx: bool = True
    onnx_opset: int = 17
    include_metadata: bool = True
    include_checksums: bool = True
    include_manifest: bool = True
    include_training_summary: bool = True


@dataclass
class TrainerConfig:
    """Root configuration — aggregates all sub-configs."""

    environment: Environment = Environment.DEVELOPMENT
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
