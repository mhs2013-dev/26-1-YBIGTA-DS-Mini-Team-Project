import torch
from torch import nn

from .blocks import TransformerBlock


class VisionEncoder(nn.Module):
    """
    CLIP의 image encoder를 작게 구현한 Vision Transformer.
    이미지를 patch token으로 바꾼 뒤 CLS token을 대표 feature로 쓴다.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        num_patches = (image_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, width))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, heads, mlp_ratio, dropout) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)

        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [batch, 3, image_size, image_size]
        x = self.patch_embed(images)
        # x: [batch, width, grid, grid]

        x = x.flatten(2).transpose(1, 2)
        # x: [batch, num_patches, width]

        cls = self.class_embedding.expand(images.size(0), -1, -1)
        # cls: [batch, 1, width]

        x = torch.cat([cls, x], dim=1)
        # x: [batch, 1 + num_patches, width]

        x = self.dropout(x + self.positional_embedding)
        # x: [batch, 1 + num_patches, width]

        for block in self.blocks:
            x = block(x)
        # x: [batch, 1 + num_patches, width]

        cls_feature = self.ln_post(x[:, 0])
        # cls_feature: [batch, width]
        return cls_feature
