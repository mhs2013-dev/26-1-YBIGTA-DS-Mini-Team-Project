import torch
from torch import nn

from .blocks import TransformerBlock
from .tokenizer import SimpleTokenizer


class TextEncoder(nn.Module):
    """
    CLIP의 text encoder를 작게 구현한 Transformer.
    마지막 EOS token 위치의 feature를 문장 대표 feature로 사용한다.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.zeros(1, context_length, width))
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, heads, mlp_ratio, dropout) for _ in range(layers)]
        )
        self.ln_final = nn.LayerNorm(width)

        nn.init.normal_(self.positional_embedding, std=0.01)

    def build_causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        # mask: [text_length, text_length]
        # True인 위치는 attention에서 보지 못한다.
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, text_length]
        text_length = input_ids.size(1)

        x = self.token_embedding(input_ids)
        # x: [batch, text_length, width]

        x = x + self.positional_embedding[:, :text_length]
        # x: [batch, text_length, width]

        causal_mask = self.build_causal_mask(text_length, input_ids.device)
        # causal_mask: [text_length, text_length]

        padding_mask = input_ids.eq(SimpleTokenizer.pad_token_id)
        # padding_mask: [batch, text_length]

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=padding_mask)
        # x: [batch, text_length, width]

        x = self.ln_final(x)
        # x: [batch, text_length, width]

        eos_positions = input_ids.eq(SimpleTokenizer.eos_token_id).int().argmax(dim=1)
        # eos_positions: [batch]

        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_feature = x[batch_idx, eos_positions]
        # sentence_feature: [batch, width]
        return sentence_feature
