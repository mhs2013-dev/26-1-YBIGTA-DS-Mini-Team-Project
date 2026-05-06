# CLIP 미니 구현 설명서

이 폴더는 CLIP 논문의 핵심 구조를 작게 줄여 직접 구현한 코드입니다. 목적은 학습 성능이 아니라 구조 이해입니다.

CLIP의 핵심은 단순합니다.

```text
이미지 encoder가 이미지 embedding을 만든다.
텍스트 encoder가 텍스트 embedding을 만든다.
맞는 이미지-텍스트 쌍은 가깝게, 틀린 쌍은 멀게 만든다.
```

---

## 폴더 구조

```text
CLIP/
├─ main.py
├─ clip paper.pdf
├─ clip_core/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ losses.py
│  ├─ model.py
│  └─ modules/
│     ├─ __init__.py
│     ├─ blocks.py
│     ├─ tokenizer.py
│     ├─ vision_encoder.py
│     └─ text_encoder.py
└─ docs/
   └─ README.md
```

| 파일 | 역할 |
|---|---|
| `main.py` | 더미 데이터로 CLIP-mini를 한 번 실행합니다. |
| `clip_core/config.py` | 모델 크기, 이미지 크기, 텍스트 길이 같은 설정값을 둡니다. |
| `clip_core/data.py` | 랜덤 이미지와 예시 문장을 만듭니다. |
| `clip_core/model.py` | 이미지 encoder, 텍스트 encoder, similarity, loss를 연결합니다. |
| `clip_core/losses.py` | CLIP contrastive loss를 계산합니다. |
| `clip_core/modules/vision_encoder.py` | 이미지를 patch token으로 바꾸고 대표 feature를 만듭니다. |
| `clip_core/modules/text_encoder.py` | 문장을 token embedding으로 바꾸고 대표 feature를 만듭니다. |
| `clip_core/modules/blocks.py` | Transformer block과 MLP입니다. |
| `clip_core/modules/tokenizer.py` | 외부 tokenizer 없이 더미 실행용 token id를 만듭니다. |

---

## 실행하기

```powershell
cd CLIP
python main.py
```

예상 출력은 이런 형태입니다.

```text
CLIP-mini dummy run | images=(4, 3, 64, 64) | text=(4, 32) | logits=(4, 4) | loss=...
sample text: a photo of a dog
```

---

## CLIP이 BLIP과 다른 점

BLIP은 이미지와 텍스트를 cross-attention으로 직접 섞는 구조가 들어갑니다.

CLIP은 더 단순합니다.

```text
이미지 encoder 따로
텍스트 encoder 따로
마지막 embedding끼리만 비교
```

즉, CLIP에서는 이미지 token과 텍스트 token이 중간에서 직접 섞이지 않습니다. 마지막에 나온 두 벡터의 유사도만 계산합니다.

---

## main.py 흐름

```python
config = CLIPConfig()
model = CLIPMini(config)
images, input_ids, raw_texts = make_dummy_batch(config)
out = model(images, input_ids)
```

`out`에는 다음 값들이 들어 있습니다.

```text
image_embeddings: [batch, projection_dim]
text_embeddings: [batch, projection_dim]
logits_per_image: [batch, batch]
logits_per_text: [batch, batch]
loss: scalar
```

---

## VisionEncoder

입력 이미지는 다음 shape입니다.

```text
images: [batch, 3, image_size, image_size]
```

현재 설정에서는:

```text
image_size = 64
patch_size = 16
```

따라서 이미지는 16개 patch로 나뉩니다.

```text
64 / 16 = 4
4 * 4 = 16 patches
```

CLS token 1개를 앞에 붙이므로 transformer에 들어가는 token 수는 17개입니다.

```text
[batch, 17, width]
```

마지막에는 CLS token 위치의 값을 이미지 대표 feature로 씁니다.

```text
cls_feature: [batch, width]
```

---

## TextEncoder

문장은 먼저 token id가 됩니다.

```text
"a photo of a dog"
→ [BOS, ..., EOS, PAD, PAD]
```

입력 shape는 다음입니다.

```text
input_ids: [batch, text_length]
```

Transformer를 통과하면:

```text
x: [batch, text_length, width]
```

CLIP에서는 문장 끝 token인 EOS 위치의 feature를 문장 대표 feature로 사용합니다.

```text
sentence_feature: [batch, width]
```

---

## Similarity matrix

이미지 embedding과 텍스트 embedding은 각각 normalize됩니다.

```text
image_embeddings: [batch, projection_dim]
text_embeddings: [batch, projection_dim]
```

그리고 행렬곱을 합니다.

```python
logits_per_image = image_embeddings @ text_embeddings.T
```

결과는 다음 shape입니다.

```text
logits_per_image: [batch, batch]
```

예를 들어 batch가 4이면:

```text
             text0 text1 text2 text3
image0         ?     ?     ?     ?
image1         ?     ?     ?     ?
image2         ?     ?     ?     ?
image3         ?     ?     ?     ?
```

정답은 대각선입니다.

```text
image0-text0
image1-text1
image2-text2
image3-text3
```

---

## Contrastive loss

CLIP loss는 두 방향으로 계산합니다.

```text
image -> text 분류
text -> image 분류
```

그리고 둘을 평균냅니다.

```python
loss = (image_loss + text_loss) / 2
```

이 loss가 시키는 일은 다음입니다.

```text
맞는 이미지-텍스트 쌍은 similarity를 크게
틀린 이미지-텍스트 쌍은 similarity를 작게
```

---

## 일부러 단순화한 부분

| 실제 CLIP | 현재 구현 |
|---|---|
| 대규모 이미지-텍스트 데이터셋 | 더미 이미지와 예시 문장 |
| BPE tokenizer | 문자 단위 SimpleTokenizer |
| 큰 ResNet/ViT encoder | 작은 ViT encoder |
| 실제 학습 루프 | 제외 |
| mixed precision/distributed training | 제외 |

이 구현은 논문을 완전히 재현하기보다, 핵심 구조를 눈으로 따라갈 수 있게 만든 버전입니다.
