# Vision-Language Model Mini Implementation

YBIGTA DS 미니프로젝트로 진행한 Vision-Language Model 논문 구현 프로젝트입니다.

이 저장소의 목표는 논문 성능을 그대로 재현하는 것이 아니라, 이미지와 텍스트를 함께 다루는 모델이 코드 안에서 어떤 구조로 움직이는지 직접 확인하는 것입니다. 그래서 대규모 학습, pretrained weight, 실제 데이터셋 연결보다는 핵심 모듈을 작은 단위로 나누고, 더미 데이터로 forward pass가 실제로 돌아가게 만드는 데 초점을 두었습니다.

현재 구현한 모델은 두 가지입니다.

- `CLIP`: 이미지 encoder와 텍스트 encoder를 따로 두고, 두 embedding의 유사도를 contrastive loss로 맞추는 구조
- `BLIP`: 이미지 encoder, text encoder, image-grounded encoder/decoder, ITC/ITM/LM objective를 포함한 구조

---

## Project Motivation

이번 프로젝트는 “논문을 읽었다”에서 끝내지 않고, 논문 속 구조가 실제 코드로 어떻게 나뉘는지 이해하기 위해 시작했습니다.

특히 아래 질문에 답할 수 있게 만드는 것이 목적입니다.

- 이미지는 어떻게 patch token으로 바뀌는가?
- 텍스트는 어떻게 token id와 embedding으로 바뀌는가?
- 이미지와 텍스트는 같은 벡터 공간에서 어떻게 비교되는가?
- CLIP의 contrastive loss는 정확히 무엇을 맞추는가?
- BLIP에서는 이미지와 텍스트가 cross-attention으로 어떻게 만나는가?
- ITC, ITM, LM objective는 코드에서 각각 어디에 대응되는가?

초보자가 코드를 따라가며 이해할 수 있도록 tensor를 다루는 주요 지점에는 shape 주석을 달고, 모델을 기능별 파일로 나누었습니다.

---

## Repository Structure

```text
.
├─ CLIP/
│  ├─ main.py
│  ├─ clip paper.pdf
│  ├─ clip_core/
│  │  ├─ config.py
│  │  ├─ data.py
│  │  ├─ losses.py
│  │  ├─ model.py
│  │  └─ modules/
│  │     ├─ blocks.py
│  │     ├─ tokenizer.py
│  │     ├─ vision_encoder.py
│  │     └─ text_encoder.py
│  └─ docs/
│     └─ README.md
│
├─ BLIP/
│  ├─ main.py
│  ├─ blip paper.pdf
│  ├─ blip/
│  │  ├─ config.py
│  │  ├─ data.py
│  │  ├─ losses.py
│  │  ├─ model.py
│  │  └─ modules/
│  │     ├─ blocks.py
│  │     ├─ vision.py
│  │     └─ text.py
│  └─ docs/
│     ├─ README.md
│     └─ blip_beginner_guide.html
│
└─ README.md
```

---

## CLIP Mini

CLIP은 구조가 비교적 단순합니다.

```text
image -> image encoder -> image embedding
text  -> text encoder  -> text embedding
image embedding과 text embedding의 similarity matrix 계산
contrastive loss로 정답 쌍은 가깝게, 오답 쌍은 멀게 학습
```

현재 구현에서는 작은 Vision Transformer와 작은 Text Transformer를 직접 만들었습니다. 실제 CLIP처럼 거대한 데이터셋을 쓰지는 않고, 랜덤 이미지 tensor와 예시 문장을 사용해 모델 흐름만 확인합니다.

실행:

```powershell
cd CLIP
python main.py
```

예상 출력:

```text
CLIP-mini dummy run | images=(4, 3, 64, 64) | text=(4, 32) | logits=(4, 4) | loss=...
sample text: a photo of a dog
```

자세한 설명은 `CLIP/docs/README.md`에 정리했습니다.

---

## BLIP Mini

BLIP은 CLIP보다 조금 더 복합적인 구조입니다. 이미지와 텍스트를 단순히 마지막 embedding에서만 비교하는 것이 아니라, 텍스트가 이미지 token을 cross-attention으로 참고하는 흐름이 들어갑니다.

현재 구현에는 다음 요소가 들어 있습니다.

- Image Encoder: 작은 Vision Transformer
- Text Encoder: 텍스트 token을 읽는 Transformer
- Image-grounded Text Encoder: 텍스트가 이미지 token을 참고하는 cross-attention 구조
- Image-grounded Text Decoder: 이미지를 참고해 다음 token을 예측하는 causal 구조
- Losses: ITC, ITM, LM

실행:

```powershell
cd BLIP
python main.py
```

예상 출력:

```text
BLIP-mini dummy run | images=(2, 3, 64, 64) | text=(2, 32) | image_tokens=(2, 17, 128) | ITC=..., ITM=..., LM=..., total=...
sample text: a small cat on the desk
```

자세한 설명은 `BLIP/docs/README.md`와 `BLIP/docs/blip_beginner_guide.html`에 정리했습니다.

---

## What Is Simplified

이 프로젝트는 논문 전체를 완전 재현하는 구현이 아닙니다. 구조 이해를 위해 아래 부분을 의도적으로 줄였습니다.

| 실제 논문 구현 | 이 프로젝트 |
|---|---|
| 대규모 이미지-텍스트 데이터셋 | 더미 이미지 tensor와 예시 문장 |
| pretrained encoder | 작은 Transformer 직접 구현 |
| BPE/BERT tokenizer | 문자 단위 간단 tokenizer |
| 긴 학습 루프 | forward pass와 loss 계산 확인 |
| distributed training | 제외 |
| 성능 평가 | 제외 |

이렇게 줄인 이유는 처음부터 실제 논문 코드 수준의 복잡도를 따라가면, 핵심 구조보다 데이터 처리와 학습 환경에 먼저 막히기 때문입니다.

---

## Study Order

처음 보는 경우에는 아래 순서로 읽는 것을 추천합니다.

1. `CLIP/main.py` 또는 `BLIP/main.py`
2. 각 모델의 `config.py`
3. 각 모델의 `data.py`
4. image encoder 파일
5. text encoder 파일
6. `model.py`
7. `losses.py`
8. `docs/README.md`

코드를 읽을 때는 함수 이름보다 tensor shape를 먼저 따라가는 것이 좋습니다. 예를 들어 `images`가 `[batch, 3, height, width]`에서 시작해 `[batch, projection_dim]`까지 어떻게 바뀌는지 보면 모델 흐름이 훨씬 잘 보입니다.

---

## Environment

현재 코드는 PyTorch 기반입니다.

```powershell
python main.py
```

형태로 각 모델 폴더 안에서 바로 실행할 수 있게 작성했습니다. 별도의 학습 데이터 다운로드는 필요하지 않습니다.

---

## Contributor

- Organization: YBIGTA
- Track: Data Science
- Project Type: Paper implementation mini project

감사합니다.