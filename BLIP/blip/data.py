import torch

# 동일한 패키지에서의 접근이기에 .을 붙여서 사용
# modules에서는 내부에 __init__.py가 있어서 from .modules로 접근 가능
from .config import BLIPConfig
from .modules import SimpleTokenizer

# make_dummy_batch 함수는 모델에 입력으로 사용할 더미 데이터를 생성하는 함수입니다.
def make_dummy_batch(config: BLIPConfig) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    texts = [
        "a small cat on the desk",
        "a person riding a bike",
    ]
    tokenizer = SimpleTokenizer(config.vocab_size, config.max_text_len)
    input_ids = tokenizer.encode(texts)

    # 2장의 3개 채널을 지닌(보편적인 RGB 이미지) 더미 이미지를 생성합니다. 이미지의 크기는 config.image_size x config.image_size입니다.
    # 아마 64 * 64 크기의 이미지가 16 * 16 크기의 패치로 나뉘어서 모델에 입력으로 들어갈 것입니다.
    images = torch.randn(len(texts), 3, config.image_size, config.image_size)
    return images, input_ids, texts
