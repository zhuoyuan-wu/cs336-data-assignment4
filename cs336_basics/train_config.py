from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    train_bin: Path
    valid_bin: Path
    model_output: Path


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 512
    d_model: int = 1024
    d_ff: int = 3072
    num_layers: int = 24
    num_heads: int = 16
    rope_theta: float = 10000.0


@dataclass
class TrainingConfig:
    seed: int = 0
    dtype: str = "bfloat16"

    # 单卡 4090D 极致安全显存配置 (~10GB)
    train_batch_size: int = 16
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 16  # 数学等价 Global Batch Size = 256

    # 严格 2 小时物理红线: 1,300 步
    train_steps: int = 1_300

    compile: bool = True

    # 高频评测点
    eval_interval: int = 65
    eval_iterations: int = 30

    max_grad_norm: float | None = 1.0
    device: str = "cuda"
    lr: float = 6e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    wandb_project: str | None = None
    wandb_entity: str | None = None
    log_interval: int = 10
    save_checkpoints: bool = False


@dataclass
class Config:
    paths: PathsConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
