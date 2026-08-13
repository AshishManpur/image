from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class DatasetConfig:
    data_root: Path = PROJECT_ROOT / "Data" / "train" / "train"
    test_root: Path = PROJECT_ROOT / "Data" / "Test_NoisyLR (1)"
    input_dir_name: str = "NoisyLR"
    target_dir_name: str = "GT"
    validation_split: float = 0.1
    seed: int = 42
    patch_size: int = 128
    use_augmentation: bool = True
    preload: bool = False
    normalization: str = "percentile"  # "none", "mean_std", "percentile"
    mean: float | None = None
    std: float | None = None
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    percentile_sample_pixels: int = 4096
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(slots=True)
class AugmentationConfig:
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.2
    rotation_prob: float = 0.5
    random_crop_prob: float = 1.0
    brightness_prob: float = 0.3
    contrast_prob: float = 0.3
    gamma_prob: float = 0.2
    gaussian_noise_prob: float = 0.4
    speckle_noise_prob: float = 0.4
    poisson_noise_prob: float = 0.2
    gaussian_blur_prob: float = 0.3
    motion_blur_prob: float = 0.2
    rotation_choices: tuple[int, ...] = (0, 90, 180, 270)
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    gamma_range: tuple[float, float] = (0.85, 1.15)
    noise_std_range: tuple[float, float] = (0.001, 0.02)
    blur_kernel_range: tuple[int, int] = (3, 7)
    motion_kernel_size: int = 7


@dataclass(slots=True)
class ModelConfig:
    in_channels: int = 1
    out_channels: int = 1
    embed_dim: int = 48
    scale_factor: int = 2
    depths: tuple[int, int, int, int] = (2, 3, 4, 4)
    num_heads: tuple[int, int, int, int] = (1, 2, 4, 8)
    mlp_ratio: float = 2.66
    dropout: float = 0.0
    use_checkpointing: bool = False


@dataclass(slots=True)
class LossConfig:
    l1_weight: float = 1.0
    ssim_weight: float = 0.5
    perceptual_weight: float = 0.1
    edge_weight: float = 0.1
    use_perceptual: bool = False
    use_edge: bool = True
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    cosine_min_lr: float = 1e-6
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    amp: bool = True
    compile: bool = True
    save_every_epoch: bool = True
    log_dir: Path = PROJECT_ROOT / "outputs" / "logs"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    output_dir: Path = PROJECT_ROOT / "outputs"
    resume_from: Path | None = None
    device: str = "auto"

    @property
    def compile_model(self) -> bool:
        return self.compile

    @compile_model.setter
    def compile_model(self, value: bool) -> None:
        self.compile = value


@dataclass(slots=True)
class ExportConfig:
    enable_onnx: bool = True
    enable_tensorrt: bool = False
    onnx_opset: int = 17


@dataclass(slots=True)
class ProjectConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def build_default_config() -> ProjectConfig:
    return ProjectConfig()
