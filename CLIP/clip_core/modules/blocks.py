import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, width: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(width * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, sequence_length, width]
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP(width, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [batch, sequence_length, width]
        h = self.ln_1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x
