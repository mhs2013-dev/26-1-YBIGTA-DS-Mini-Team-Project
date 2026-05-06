import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import CLIPConfig
from .losses import clip_contrastive_loss
from .modules.text_encoder import TextEncoder
from .modules.vision_encoder import VisionEncoder


class CLIPMini(nn.Module):
    """
    CLIP의 핵심만 남긴 작은 dual encoder.

    image -> VisionEncoder -> image embedding
    text  -> TextEncoder   -> text embedding
    두 embedding의 similarity matrix를 만들고 contrastive loss를 계산한다.
    """

    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.config = config

        self.visual = VisionEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            width=config.width,
            layers=config.vision_layers,
            heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
        self.text = TextEncoder(
            vocab_size=config.vocab_size,
            context_length=config.max_text_len,
            width=config.width,
            layers=config.text_layers,
            heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

        self.image_projection = nn.Linear(config.width, config.projection_dim, bias=False)
        self.text_projection = nn.Linear(config.width, config.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(config.init_logit_scale))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # images: [batch, 3, image_size, image_size]
        image_features = self.visual(images)
        # image_features: [batch, width]
        image_embeddings = self.image_projection(image_features)
        # image_embeddings: [batch, projection_dim]
        return F.normalize(image_embeddings, dim=-1)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, text_length]
        text_features = self.text(input_ids)
        # text_features: [batch, width]
        text_embeddings = self.text_projection(text_features)
        # text_embeddings: [batch, projection_dim]
        return F.normalize(text_embeddings, dim=-1)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids)

        # logit_scale: scalar. 논문에서는 exp(logit_scale)을 similarity에 곱한다.
        logit_scale = self.logit_scale.exp().clamp(max=100)

        # logits_per_image: [batch, batch]
        # row i, col j = i번째 이미지와 j번째 텍스트의 유사도 점수
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()

        # logits_per_text: [batch, batch]
        # row i, col j = i번째 텍스트와 j번째 이미지의 유사도 점수
        logits_per_text = logits_per_image.t()

        loss = clip_contrastive_loss(logits_per_image, logits_per_text)

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "logit_scale": logit_scale,
            "loss": loss,
        }
