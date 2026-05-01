import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 자기 자신끼리 셀프 어텐션하는 구조의 트랜스포머 블록입니다. 텍스트 트랜스포머와 비전 트랜스포머에서 공통으로 사용됩니다.
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, _ = self.self_attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + h
        return x + self.mlp(self.norm2(x))

# 텍스트가 이미지리를 바라보게 하는 크로스 어텐션 블록입니다. 텍스트 트랜스포머에서 이미지 토큰이 주어지는 경우에 사용됩니다.
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, text: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        h, _ = self.cross_attn(
            self.norm_q(text),
            self.norm_kv(image_tokens),
            self.norm_kv(image_tokens),
            need_weights=False,
        )
        text = text + h
        return text + self.mlp(self.norm_mlp(text))
