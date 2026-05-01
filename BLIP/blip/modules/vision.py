import torch
from torch import nn

from .blocks import TransformerBlock


class VisionTransformer(nn.Module):
    """BLIP의 image encoder 자리에 들어가는 작은 ViT."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None: # 여기서 논은 이 함수의 변환값이 없음을 시사함
        # 부모 클래스를 초기화하여 내 레이어를 등록할 수 있게 할 거임.
        super().__init__()
        # 못 받아들일거라면 에러를 방출함.
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        # 패치를 단위로 잘라서 임배딩 벡터 형태로 바꾸겠다.
        # 현재 입력 이미지(원본)은 3, 64, 64 형태이지만 패치 사이즈가 16이므로 패치로 나누면 4, 4개의 패치가 생기고 각 패치는 3, 16, 16 형태입니다. 이걸 임배딩 벡터로 바꿔주는 레이어입니다.
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        # 맨 앞에 [CLS] token을 위한 임베딩 벡터와 위치 임베딩 벡터를 초기화합니다. [CLS] 토큰은 이미지 전체의 표현을 담는 역할을 합니다.
        # 이건 당연히 학습해야 하니까 parameter로 등록합니다.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 위치 임배딩을 할 예정. 토큰에 대한 위치 정보를 반영하기 위함.
        # num_patches + 1인 이유는 [CLS] 토큰도 위치 임베딩이 필요하기 때문입니다.
        # 위치 임베딩도 학습해야 하니까 parameter로 등록합니다.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # dropout도 레이어로 등록합니다. (보편적으로 과적합 방지용)
        self.dropout = nn.Dropout(dropout)
        # 트렌스포머 블록을 depth 수만큼 쌓아서 모델의 깊이를 만듭니다. 각 블록은 embed_dim 차원의 벡터를 입력으로 받아서 같은 차원의 벡터를 출력합니다.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # 위에서 정의한 토큰들에 대해 (초기에는 0으로 시작했지만) 적당히 초기화를 시켜 줍니다. 
        #  보통은 작은 정규분포에서 초기화하는 경우가 많습니다. (여기에선 절단 정규분포로 처리)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 이미지를 patch 임배딩으로 변환
        # (B, 3, 64, 64) -> (B, embed_dim, 4, 4) 형태가 됩니다. 여기서 B는 배치 사이즈입니다.
        # 우리는 앞으로 (B, 128, 4, 4) 형태의 텐서를 다루게 될 것입니다.
        x = self.patch_embed(images)
        # flatten과 transpose를 이용해서 (B, embed_dim, 4, 4) -> (B, num_patches, embed_dim) 형태로 바꿔줍니다. 여기서 num_patches는 16입니다.
        # 즉, (B, 16, 128) 형태가 됩니다. 각 패치가 하나의 토큰으로 표현되고, 각 토큰은 embed_dim 차원의 벡터로 표현됩니다.
        x = x.flatten(2).transpose(1, 2)
        # CLS token을 batch 크기만큼 복사 (각 배치 속 이미지마다 1개가 필요하기에)
        # -1 은 그냥 그 자리의 차원 크기를 유지하라는 뜻입니다. 즉, (B, 1, embed_dim) 형태가 됩니다.
        cls = self.cls_token.expand(images.size(0), -1, -1)
        # CLS 토큰을 패치 임베딩 시퀀스의 맨 앞에 붙입니다. 이렇게 하면 (B, num_patches + 1, embed_dim) 형태가 됩니다.
        x = torch.cat([cls, x], dim=1)
        # 위치 임배딩을 더한다.
        # 근데 이 과정에서 브로드 케스팅 과정이 일어난다.
        # pos_embed의 형태는 (1, num_patches + 1, embed_dim)인데 x의 형태는 (B, num_patches + 1, embed_dim)이기 때문에 pos_embed가 배치 크기 B에 맞게 자동으로 확장되어 더해집니다.
        # 자동으로 반복되는건데, 위치 임배딩은 이미지(배치 속 요소)마다 다를 필요가 없기에, 반복해서 더해줘도 괜찮다.
        x = self.dropout(x + self.pos_embed)
        #안의 트렌스포머 블럭을 모조리 통과한다.
        # 블럭 안에서 토큰들끼리 self-attention을 통해 서로의 정보를 주고받으면서 점점 더 풍부한 표현으로 변환됩니다.
        for block in self.blocks:
            x = block(x)
        # 정규화하면서 마무리
        return self.norm(x)
