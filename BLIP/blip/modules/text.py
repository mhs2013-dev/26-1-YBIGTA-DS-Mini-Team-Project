from __future__ import annotations

import torch
from torch import nn

from .blocks import CrossAttentionBlock, TransformerBlock

# 진짜 문자 단위로 사용되는 임시 토크나이저, 'abc' -> [bos, a, b, c, eos, pad, pad, ...] 이런 식으로 토큰 아이디로 변환해주는 역할을 합니다.
class SimpleTokenizer:
    """외부 토크나이저 없이 더미 실행을 하기 위한 문자 단위 토크나이저."""
    # padding token
    pad_token_id = 0
    # beginning of sentence token
    bos_token_id = 1
    # end of sentence token
    eos_token_id = 2
    # unknown token (for characters that are not in the vocab)
    unk_token_id = 3

    def __init__(self, vocab_size: int, max_len: int) -> None:
        if vocab_size < 16:
            raise ValueError("vocab_size should be at least 16")
        self.vocab_size = vocab_size
        self.max_len = max_len
    # 문자열 리스트를 받아서 텐서 형태로 바꾸는 함수
    # 출략은 (batch_size, max_len) 형태의 텐서가 됩니다. 
    # 각 요소는 토큰 아이디입니다.
    def encode(self, texts: list[str]) -> torch.Tensor:
        # 각 문자를 숫자 ID list로 바꾼 뒤 아래 rows에 저장한다.
        rows: list[list[int]] = []
        for text in texts:
            # 가장 앞에 bos 토큰을 넣는다.
            ids = [self.bos_token_id]
            # 문장을 소문자로 바꾼 뒤 max_len - 2 만큼의 문자에 대해서만 반복한다. (bos와 eos 토큰 때문에 2를 뺌)
            for ch in text.lower()[: self.max_len - 2]:
                # 서로 다른 값을 가져야 해서 임시적으로 ord(ch) % (self.vocab_size - 4) + 4로 토큰 아이디를 계산한다. (0, 1, 2, 3은 특별 토큰이니까 그 뒤부터 시작)
                # 앞 뒤에서 +- 4로 해서 결국 vocab_size 범위 내에서 순환하는 형태가 됩니다. (예: vocab_size=128이면 4~127 사이에서 순환)
                ids.append(4 + (ord(ch) % (self.vocab_size - 4)))
            # 마지막을 의미하는 토큰을 붙인다
            ids.append(self.eos_token_id)
            # max_len보다 짧으면 패딩 토큰으로 채운다. ids 리스트의 길이가 max_len이 되도록 pad_token_id를 반복해서 추가한다.
            ids += [self.pad_token_id] * (self.max_len - len(ids))
            # 이제 이걸 저장한다. ids 리스트는 이제 [bos, token1, token2, ..., eos, pad, pad, ...] 형태가 됩니다. 길이는 max_len입니다.
            rows.append(ids[: self.max_len])
            # 텐서 형태로 바꿔서 반환한다. (batch_size, max_len) 형태의 텐서가 됩니다. 각 요소는 토큰 아이디입니다.
        return torch.tensor(rows, dtype=torch.long)

# 위에서는 텍스트를 토큰 아이디로 바꿔주는 역할을 하는 SimpleTokenizer 클래스를 정의했습니다.
# 이제는 각 토큰에 백터를 할당하는 임베딩 레이어와 트랜스포머 블록을 쌓아서 텍스트를 처리하는 TextTransformer 클래스를 정의합니다.
class TextTransformer(nn.Module):
    """text encoder, image-grounded encoder, caption decoder를 한 몸으로 쓴다."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        # 셀프 어텐션 하는 부분
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        # 이미지 토큰과 크로스 어텐션 하는 부분 (이미지 토큰이 주어지는 경우에만 실행)
        # 텍스트가 이미지를 바라보게 한다고 생각하면 될 듯
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # 언어 모델링 헤드
        # 각 토큰 위치의 HIDDEN_DIM 차원의 벡터를 VOCAB_SIZE 차원의 로짓으로 바꿔주는 선형 레이어입니다. 
        # 이렇게 하면 각 토큰 위치에서 어떤 단어가 나올지 예측할 수 있습니다.
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        # 위치 임배딩 초기화해서 학습 안정성 제공
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    # 다음 위치의 토큰들을 바라보지 못하게 마스킹하는 작업
    def causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
    # 실제 텍스트를 트렌스포머에 통과시키는 함수
    def forward(
        self,
        input_ids: torch.Tensor,
        image_tokens: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        # (batch_size, seq_len) 형태의 input_ids 텐서를 받아서 (batch_size, seq_len, embed_dim) 형태의 텐서로 바꿔주는 과정입니다.
        seq_len = input_ids.size(1)
        x = self.token_embed(input_ids) + self.pos_embed[:, :seq_len]
        # 패딩 마스크의 위치를 찾아, 어텐션하는 과정에서 이들을 연산하지 않게 함. 패딩 토큰이 있는 위치는 True가 됩니다.
        padding_mask = input_ids.eq(SimpleTokenizer.pad_token_id)
        # 인과적 (NTP) 마스크를 만들어서, causal이 True인 경우에만 다음 위치의 토큰들을 바라보지 못하게 합니다. (예: seq_len=5이면 5x5 텐서에서 upper triangular 부분이 True가 됩니다.)
        attn_mask = self.causal_mask(seq_len, input_ids.device) if causal else None
        # 위에서 정의한대로 모델이 self-attention과 cross-attention을 수행하도록 합니다.
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=padding_mask)
        # 이미지 토큰이 주어진 경우에 대해서는 크로스 어텐션을 수행한다
        if image_tokens is not None:
            # 블럭만큼 수행하는데, 여기서 텍스트 토큰이 이미지 토큰을 바라봅니다.
            for block in self.cross_blocks:
                x = block(x, image_tokens)
        # 마지막으로 정규화를 하고 내보냅니다.
        return self.norm(x)
    # 언어를 추론하기 위해 점수를 낼건데, 이 과정입니다.
    def logits(
        self,
        input_ids: torch.Tensor,
        image_tokens: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        return self.lm_head(self.forward(input_ids, image_tokens, causal))
# 포워드 과정을 통해서 히든 스테이트를 얻는다
# 그 히든 스테이트를 lm_head에 통과시켜서 각 위치에서 다음 단어가 무엇일지 예측하는 로짓을 얻는다. (batch_size, seq_len, vocab_size) 형태의 텐서가 됩니다.