import torch
from torch import nn

from .config import BLIPConfig

#losses에서 정의한 3가지의 서로 다른 손실함수를 가져온다.
from .losses import (
    image_text_contrastive_loss,
    image_text_matching_loss,
    language_modeling_loss,
)
from .modules import TextTransformer, VisionTransformer
from typing import Any
#이걸 model의 인자로 넣으면 자동으로 forward 함수에서 사용할 수 있게 됩니다.
class BLIPMini(nn.Module):
    """
    BLIP 논문의 핵심 흐름만 살린 미니 구현.
    - image encoder: ViT
    - text encoder: unimodal text representation
    - image-grounded text encoder: ITM에 쓰는 cross-attention
    - image-grounded text decoder: caption LM에 쓰는 causal self-attention + cross-attention
    """

    def __init__(self, config: BLIPConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = VisionTransformer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.vision_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
        self.text = TextTransformer(
            vocab_size=config.vocab_size,
            max_len=config.max_text_len,
            embed_dim=config.embed_dim,
            depth=config.text_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

        # 이미지와 텍스트를 동일한 공간에서 대조학습을 시키기 위한 선형 레이어입니다. 
        # ViT에서 나온 이미지 표현과 텍스트 트랜스포머에서 나온 텍스트 표현을 같은 차원으로 바꿔주는 역할을 합니다.
        self.vision_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.text_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.itm_head = nn.Linear(config.embed_dim, 2)

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_tokens = self.visual_encoder(images)
        image_cls = self.vision_proj(image_tokens[:, 0])
        # 토큰은 크로스 어텐션에서 사용할 예정이며 cls 토큰은 대조학습과 ITM에서 사용할 예정입니다. 
        # 차원은 토큰은 B, N+1, D 형태이고 cls 토큰은 B, D 형태입니다. 여기서 N은 패치의 개수입니다.
        return image_tokens, image_cls

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        text_tokens = self.text(input_ids)
        return self.text_proj(text_tokens[:, 0])

    # forawrd에서는 이미지와 텍스트를 전부 받아서 출력을 연산할거임
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> dict[str, Any]:
        image_tokens, image_feat = self.encode_image(images)
        text_feat = self.encode_text(input_ids)

        grounded_text = self.text(input_ids, image_tokens=image_tokens)

        # grounded_text[:, 0] 는 1번째 토큰의 위치를 의미한다.
        # ITM 헤드는 이미지와 텍스트가 매칭되는지 여부를 예측하는 선형 레이어입니다. grounded_text[:, 0]는 [CLS] 토큰의 위치에 해당하는 벡터로, 이미지와 텍스트의 전체적인 표현을 담고 있습니다. 이 벡터를 itm_head에 통과시켜서 매칭 여부를 예측하는 로짓을 얻습니다. 로짓의 차원은 (batch_size, 2)로, 각 샘플에 대해 매칭과 비매칭의 점수를 나타냅니다.
        # 각 이미지 - 텍스트 쌍이 matching인지 아닌지 예측하는 점수입니다. ITM은 Image-Text Matching의 약자로, 이미지와 텍스트가 서로 매칭되는지를 판단하는 작업입니다. grounded_text[:, 0]는 텍스트 트랜스포머의 출력에서 [CLS] 토큰에 해당하는 벡터를 의미합니다. 이 벡터는 텍스트 전체의 표현을 담고 있으며, 이미지 토큰과의 크로스 어텐션을 통해 이미지 정보를 반영한 상태입니다. 이 벡터를 itm_head에 통과시켜서 매칭 여부를 예측하는 로짓을 얻습니다.
        itm_scores = self.itm_head(grounded_text[:, 0])

        # caption_logits는 이미지와 텍스트가 매칭된다고 가정했을 때, 텍스트 트랜스포머가 다음 단어를 예측하는 로짓입니다. causal=True로 설정해서, 각 위치에서 다음 단어만 바라보도록 합니다. 이렇게 하면 언어 모델링이 가능해집니다.
        #1. input_ids를 token embedding으로 변환 
        #2. causal self-attention 적용
        #3. image_tokens를 cross-attention으로 참고
        #4. lm_head로 vocab logits 출력
        caption_logits = self.text.logits(input_ids, image_tokens=image_tokens, causal=True)

        losses = {
            "itc": image_text_contrastive_loss(
                image_feat, text_feat, self.config.temperature
            ),
            "itm": image_text_matching_loss(itm_scores),
            "lm": language_modeling_loss(caption_logits, input_ids),
        }
        losses["total"] = losses["itc"] + losses["itm"] + losses["lm"]

        return {
            "image_tokens": image_tokens,
            "image_feat": image_feat,
            "text_feat": text_feat,
            "itm_scores": itm_scores,
            "caption_logits": caption_logits,
            "losses": losses,
        }
