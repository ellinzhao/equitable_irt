from dataclasses import dataclass


@dataclass
class ExptConfig:
    model: str = 'unet'
    batch_size: int = 32
    epochs: int = 10

    lr: float = 2e-5
    fever_prob: float = 0.5

    sl_weight: float = 1.0
    lpips_weight: float = 1.0
    mae_temp_weight: float = 1.0
    grad_clip: float = 1.0
