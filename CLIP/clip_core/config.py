from dataclasses import dataclass


@dataclass
class CLIPConfig:
    image_size: int = 64
    patch_size: int = 16
    vocab_size: int = 256
    max_text_len: int = 32

    width: int = 128
    projection_dim: int = 128
    vision_layers: int = 2
    text_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    init_logit_scale: float = 1 / 0.07

    @property
    def num_patches(self) -> int:
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side
