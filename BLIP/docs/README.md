# BLIP 미니 구현 초보자 설명서

이 프로젝트는 BLIP 논문을 그대로 대규모 학습까지 재현하는 코드가 아니라, 논문에 나오는 핵심 구조를 작게 줄여서 직접 손으로 구현한 학습용 코드입니다.

목표는 하나입니다.

> 이미지와 텍스트가 어떻게 같은 모델 안에서 만나고, 어떤 loss로 학습되는지 코드 구조로 이해하기

---

## 1. 전체 폴더 구조

```text
BLIP/
├─ main.py
├─ blip/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ losses.py
│  ├─ model.py
│  └─ modules/
│     ├─ __init__.py
│     ├─ blocks.py
│     ├─ vision.py
│     └─ text.py
└─ docs/
   ├─ README.md
   └─ blip_beginner_guide.html
```

각 파일은 역할이 다릅니다.

| 파일 | 역할 |
|---|---|
| `main.py` | 실행 시작점. 더미 이미지/문장을 만들고 모델을 한 번 돌립니다. |
| `blip/config.py` | 모델 크기, 이미지 크기, 토큰 길이 같은 설정값을 모아둡니다. |
| `blip/data.py` | 진짜 데이터셋 대신 더미 이미지와 더미 문장을 만듭니다. |
| `blip/model.py` | BLIP 전체 모델입니다. 이미지 인코더, 텍스트 인코더, loss 계산을 연결합니다. |
| `blip/losses.py` | BLIP에서 중요한 3가지 loss를 따로 모아둡니다. |
| `blip/modules/blocks.py` | Transformer에서 반복해서 쓰는 기본 블록입니다. |
| `blip/modules/vision.py` | 이미지를 patch로 쪼개서 읽는 작은 Vision Transformer입니다. |
| `blip/modules/text.py` | 문장을 token으로 읽고, 이미지 정보와 합치는 Text Transformer입니다. |

왜 이렇게 나눴냐면, 초보자가 한 파일에 모든 코드를 넣으면 “어디가 이미지 쪽이고 어디가 텍스트 쪽인지” 바로 무너지기 때문입니다.

---

## 2. 실행 흐름 한 줄 요약

`main.py`를 실행하면 아래 순서로 움직입니다.

```text
더미 이미지/문장 생성
→ BLIPMini 모델 생성
→ 이미지 encoder 통과
→ 텍스트 encoder 통과
→ 이미지와 텍스트를 cross-attention으로 결합
→ ITC / ITM / LM loss 계산
→ shape와 loss 출력
```

실행 명령은 다음입니다.

```powershell
python main.py
```

예상 출력은 대략 이런 형태입니다.

```text
BLIP-mini dummy run | images=(2, 3, 64, 64) | text=(2, 32) | image_tokens=(2, 17, 128) | ITC=..., ITM=..., LM=..., total=...
sample text: a small cat on the desk
```

loss 숫자는 랜덤 초기화 때문에 환경에 따라 조금 달라질 수 있습니다.

---

## 3. BLIP이 하려는 일

BLIP은 이미지와 텍스트를 같이 다루는 모델입니다.

예를 들면 이런 문제를 다룹니다.

- 이미지와 문장이 서로 맞는지 판단하기
- 이미지에 어울리는 설명문 만들기
- 이미지와 텍스트를 같은 embedding 공간에 놓기

이 미니 구현에서는 그중 핵심 아이디어만 가져왔습니다.

| 논문 개념 | 이 프로젝트 코드 |
|---|---|
| Image Encoder | `VisionTransformer` |
| Text Encoder | `TextTransformer` |
| Image-grounded Text Encoder | `TextTransformer(..., image_tokens=...)` |
| Image-grounded Text Decoder | `TextTransformer.logits(..., causal=True)` |
| Image-Text Contrastive Loss | `image_text_contrastive_loss` |
| Image-Text Matching Loss | `image_text_matching_loss` |
| Language Modeling Loss | `language_modeling_loss` |

---

## 4. `main.py`: 실행 시작점

`main.py`는 공부할 때 가장 먼저 보면 되는 파일입니다.

```python
config = BLIPConfig()
model = BLIPMini(config)
images, input_ids, texts = make_dummy_batch(config)
out = model(images, input_ids)
```

이 네 줄이 핵심입니다.

1. `BLIPConfig()`  
   모델 설정값을 만듭니다.

2. `BLIPMini(config)`  
   설정값을 바탕으로 모델을 만듭니다.

3. `make_dummy_batch(config)`  
   실제 데이터셋 대신 랜덤 이미지와 간단한 문장을 만듭니다.

4. `model(images, input_ids)`  
   이미지와 문장을 모델에 넣고 결과를 받습니다.

여기서 `out`은 dictionary입니다.

```python
{
    "image_tokens": ...,
    "image_feat": ...,
    "text_feat": ...,
    "itm_scores": ...,
    "caption_logits": ...,
    "losses": ...
}
```

즉, 모델 안에서 중간 결과와 loss를 모두 볼 수 있게 해둔 구조입니다.

---

## 5. `config.py`: 설정값을 따로 둔 이유

`BLIPConfig`에는 이런 값들이 있습니다.

```python
image_size = 64
patch_size = 16
vocab_size = 128
max_text_len = 32
embed_dim = 128
vision_layers = 2
text_layers = 2
num_heads = 4
```

이 값들을 모델 코드 안에 직접 박아두면 나중에 실험하기 어렵습니다.

예를 들어 이미지 크기를 64에서 128로 바꾸고 싶을 때, 여러 파일을 뒤질 필요 없이 `config.py`만 보면 됩니다.

초보자 입장에서는 `config.py`를 “모델 조절판”이라고 생각하면 됩니다.

---

## 6. `data.py`: 더미 데이터가 필요한 이유

현재 프로젝트는 훈련이 목적이 아닙니다. 그래서 COCO 같은 큰 이미지-캡션 데이터셋을 붙이지 않았습니다.

대신 `make_dummy_batch`가 다음을 만듭니다.

```python
images = torch.randn(batch, 3, image_size, image_size)
input_ids = tokenizer.encode(texts)
```

이미지는 랜덤 숫자입니다.

```text
(2, 3, 64, 64)
```

이 뜻은 다음과 같습니다.

| 차원 | 의미 |
|---|---|
| 2 | 이미지 2장 |
| 3 | RGB 채널 |
| 64 | 높이 |
| 64 | 너비 |

텍스트는 숫자 token으로 바뀝니다.

```text
"a small cat on the desk"
→ [1, 101, 36, ..., 2, 0, 0]
```

딥러닝 모델은 문자열을 직접 이해하지 못하기 때문에 문자를 숫자로 바꿔야 합니다.

---

## 7. `vision.py`: 이미지를 모델이 읽는 방식

`VisionTransformer`는 이미지를 작은 patch로 쪼갭니다.

현재 설정은 이렇습니다.

```text
image_size = 64
patch_size = 16
```

그러면 이미지는 이렇게 나뉩니다.

```text
64 x 64 이미지
→ 16 x 16 patch가 가로 4개, 세로 4개
→ 총 16개 patch
```

코드에서는 `Conv2d`를 이용해서 patch를 만듭니다.

```python
self.patch_embed = nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

여기서 `kernel_size=patch_size`, `stride=patch_size`라서 겹치지 않게 patch를 잘라냅니다.

그 다음 `[CLS] token`을 앞에 붙입니다.

```text
16개 image patch + 1개 CLS token = 17개 image token
```

그래서 실행 결과에 이런 shape가 나옵니다.

```text
image_tokens=(2, 17, 128)
```

뜻은 다음과 같습니다.

| 차원 | 의미 |
|---|---|
| 2 | batch size |
| 17 | CLS 1개 + patch 16개 |
| 128 | 각 token의 embedding 크기 |

---

## 8. `text.py`: 문장을 모델이 읽는 방식

`SimpleTokenizer`는 문자를 숫자로 바꿉니다.

중요한 token id는 다음입니다.

| token | id | 의미 |
|---|---:|---|
| PAD | 0 | 길이를 맞추기 위한 빈 칸 |
| BOS | 1 | 문장 시작 |
| EOS | 2 | 문장 끝 |
| UNK | 3 | 모르는 문자 |

예를 들어 문장은 이런 식으로 바뀝니다.

```text
"cat"
→ [BOS, c, a, t, EOS, PAD, PAD, ...]
```

`TextTransformer`는 이 token들을 embedding으로 바꾼 뒤 Transformer block에 넣습니다.

```python
x = self.token_embed(input_ids) + self.pos_embed[:, :seq_len]
```

여기서 중요한 점은 `TextTransformer`가 세 가지 역할을 같이 한다는 것입니다.

| 사용 방식 | 역할 |
|---|---|
| `self.text(input_ids)` | text encoder |
| `self.text(input_ids, image_tokens=image_tokens)` | image-grounded text encoder |
| `self.text.logits(input_ids, image_tokens=image_tokens, causal=True)` | image-grounded text decoder |

논문에서는 encoder/decoder 역할을 구분하지만, 이 미니 구현에서는 구조를 쉽게 보기 위해 하나의 class 안에서 옵션으로 동작을 나눴습니다.

---

## 9. `blocks.py`: Transformer 기본 부품

Transformer는 크게 두 가지 연산을 반복합니다.

1. Attention
2. MLP

이 프로젝트에는 두 종류의 block이 있습니다.

### TransformerBlock

자기 자신끼리 attention합니다.

```text
text token끼리 보기
image patch끼리 보기
```

코드에서는 `nn.MultiheadAttention`을 사용합니다.

```python
self.self_attn = nn.MultiheadAttention(...)
```

### CrossAttentionBlock

텍스트가 이미지를 바라보게 합니다.

```text
text token = 질문
image token = 참고 자료
```

즉, 문장의 각 token이 이미지 patch 중 어디를 봐야 할지 배우는 구조입니다.

이게 BLIP에서 이미지와 텍스트가 만나는 핵심 부분입니다.

---

## 10. `model.py`: BLIPMini 전체 연결

`BLIPMini.forward`는 가장 중요한 흐름입니다.

```python
image_tokens, image_feat = self.encode_image(images)
text_feat = self.encode_text(input_ids)

grounded_text = self.text(input_ids, image_tokens=image_tokens)
itm_scores = self.itm_head(grounded_text[:, 0])
caption_logits = self.text.logits(input_ids, image_tokens=image_tokens, causal=True)
```

하나씩 보면 다음과 같습니다.

### 1. 이미지 인코딩

```python
image_tokens, image_feat = self.encode_image(images)
```

이미지를 ViT에 넣어서 image token들을 얻습니다.

`image_tokens`는 patch별 정보 전체이고, `image_feat`는 대표 벡터입니다.

### 2. 텍스트 인코딩

```python
text_feat = self.encode_text(input_ids)
```

문장만 읽어서 텍스트 대표 벡터를 얻습니다.

### 3. 이미지-텍스트 결합

```python
grounded_text = self.text(input_ids, image_tokens=image_tokens)
```

여기서는 텍스트가 cross-attention으로 이미지를 참고합니다.

### 4. 이미지-텍스트 매칭 점수

```python
itm_scores = self.itm_head(grounded_text[:, 0])
```

이 이미지와 이 문장이 맞는 쌍인지 분류합니다.

### 5. 캡션 생성용 logits

```python
caption_logits = self.text.logits(input_ids, image_tokens=image_tokens, causal=True)
```

이미지를 참고하면서 다음 token을 예측합니다.

---

## 11. `losses.py`: 세 가지 loss

BLIP의 핵심 학습 목적은 크게 세 가지입니다.

### ITC: Image-Text Contrastive Loss

```python
image_text_contrastive_loss(...)
```

이미지 벡터와 텍스트 벡터를 같은 공간에 놓습니다.

목표는 이렇습니다.

```text
맞는 이미지-문장 쌍은 가깝게
틀린 이미지-문장 쌍은 멀게
```

### ITM: Image-Text Matching Loss

```python
image_text_matching_loss(...)
```

이미지와 문장이 실제로 맞는 쌍인지 분류합니다.

이 미니 구현에서는 더미 실행을 단순하게 하기 위해 모든 쌍을 positive로 둡니다.

진짜 학습에서는 negative pair도 같이 만들어야 합니다.

### LM: Language Modeling Loss

```python
language_modeling_loss(...)
```

이미지를 보고 다음 단어를 맞히는 loss입니다.

예를 들면 다음과 같습니다.

```text
입력: a small cat
예측: small cat on
```

정확히는 각 위치에서 “다음 token”을 맞히게 합니다.

---

## 12. 지금 구현에서 단순화한 부분

이 코드는 논문 전체 재현이 아니라 학습용 미니 구현입니다.

단순화한 부분은 다음입니다.

| 실제 BLIP | 현재 구현 |
|---|---|
| 큰 pretrained ViT | 작은 ViT 직접 구현 |
| BERT tokenizer | 문자 단위 SimpleTokenizer |
| 큰 이미지-텍스트 데이터셋 | 랜덤 이미지 + 더미 문장 |
| caption filtering, bootstrapping | 제외 |
| hard negative mining | 제외 |
| 실제 학습 루프 | 제외 |

이렇게 줄인 이유는 초보자가 처음부터 논문 전체 코드를 보면 핵심 구조를 보기 전에 데이터셋, 토크나이저, 분산학습, pretrained weight에서 막히기 때문입니다.

---

## 13. 공부 순서 추천

처음 볼 때는 이 순서가 좋습니다.

1. `main.py`  
   전체 실행 흐름을 봅니다.

2. `blip/config.py`  
   모델 크기를 정하는 값들을 봅니다.

3. `blip/data.py`  
   이미지와 문장이 어떤 shape로 들어가는지 봅니다.

4. `blip/modules/vision.py`  
   이미지가 patch token으로 바뀌는 과정을 봅니다.

5. `blip/modules/text.py`  
   문장이 token embedding으로 바뀌는 과정을 봅니다.

6. `blip/model.py`  
   이미지와 텍스트가 어디서 만나는지 봅니다.

7. `blip/losses.py`  
   세 가지 loss가 무엇을 시키는지 봅니다.

---

## 14. 코드 읽을 때 꼭 기억할 것

이 프로젝트에서 가장 중요한 문장은 이것입니다.

> 이미지는 `image_tokens`가 되고, 문장은 `input_ids`가 되며, 둘은 `cross-attention`에서 만난다.

그리고 loss는 세 가지 역할을 합니다.

```text
ITC: 이미지 벡터와 텍스트 벡터를 가깝게 만든다.
ITM: 이미지와 문장이 맞는 쌍인지 판단한다.
LM: 이미지를 참고해서 다음 token을 맞힌다.
```

이 정도를 이해하면, BLIP 논문 구조를 코드 관점에서 1차로 잡은 것입니다.
