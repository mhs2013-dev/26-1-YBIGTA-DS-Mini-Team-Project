import torch
import torch.nn.functional as F


def image_text_contrastive_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    # 정규화 시행
    # 이미지 특징과 텍스트 특징을 L2 정규화하여 단위 벡터로 만듭니다. 이렇게 하면 코사인 유사도를 계산할 때 내적이 곧 유사도가 됩니다.
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    # 전지하고 행렬곱하면 각 배치사이즈 * 배치사이즈 형태의 유사도 행렬이 나옵니다. 
    # temperature로 나눠서 스케일링합니다. temperature가 낮을수록 모델이 더 확신을 가지고 예측하도록 만듭니다.
    logits = image_feat @ text_feat.t() / temperature
    # 정답이 곧 본인 인덱스니까 (들어올떄 pair된 상태로, 메칭된 상태로 들어올테니까)
    # 이를 기반으로 이미지에서 텍스트로 예측하는 손실과 텍스트에서 이미지로 예측하는 손실을 계산합니다. 두 손실의 평균을 반환합니다.
    labels = torch.arange(logits.size(0), device=logits.device)
    # 이미지에서 텍스트를 검색하는 쪽의 방향의 로스
    loss_i2t = F.cross_entropy(logits, labels)
    # 텍스르에서 이미지를 검색하는 쪽의 방향의 로스
    loss_t2i = F.cross_entropy(logits.t(), labels)
    # 이들의 평균을 반환합니다. 이렇게 하면 모델이 이미지와 텍스트가 서로 매칭되는지를 양방향으로 학습할 수 있습니다.
    return (loss_i2t + loss_t2i) / 2

# 이 함수는 ITM loss(image-text matching loss, 이미지-텍스트 매칭 손실)를 계산합니다.
# 입력 scores는 보통 itm_head에서 나온 값입니다
def image_text_matching_loss(scores: torch.Tensor) -> torch.Tensor:
    # 스코어스는 (batch_size, 2) 형태로, 각 샘플에 대해 매칭과 비매칭의 점수를 나타냅니다. 
    # 정답 레이블은 매칭이므로 1로 설정합니다. 
    # 이렇게 하면 모델이 매칭된 이미지-텍스트 쌍에 높은 점수를 주도록 학습됩니다.
    labels = torch.ones(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, labels)


def language_modeling_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # 마지막 토큰은 그 뒤의 토큰이 없기에 예측 자체가 불가함
    # 그래서 일단 마지막꺼는 제외한다는 뜻
    pred = logits[:, :-1].contiguous()
    target = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        pred.view(-1, pred.size(-1)),
        target.view(-1),
        ignore_index=0,
    )

'''contiguous()는 PyTorch에서 텐서의 메모리 레이아웃을 연속적으로 만들어주는 메서드입니다.
PyTorch에서 텐서는 메모리에 연속적으로 저장되어야 효율적으로 연산할 수 있습니다. 하지만 어떤 연산을 수행하다 보면 텐서의 메모리 레이아웃이 비연속적이 될 수 있습니다. 예를 들어, 슬라이싱이나 트랜스포즈 같은 연산을 수행하면 텐서의 메모리 레이아웃이 비연속적이 될 수 있습니다.'''