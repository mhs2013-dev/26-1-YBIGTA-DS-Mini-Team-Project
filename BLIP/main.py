import torch

from blip import BLIPConfig, BLIPMini
from blip.data import make_dummy_batch


def main() -> None:
    torch.manual_seed(7)
    # 모델에 대한 설정을 불러옴
    config = BLIPConfig()
    # 모델을 초기화하고 평가 모드로 설정
    model = BLIPMini(config)
    model.eval()
    #실제 data를 사용할 건 아니니까 dummy batch를 만들어서 모델에 입력으로 넣어봄
    images, input_ids, texts = make_dummy_batch(config)

    with torch.no_grad():
        out = model(images, input_ids)

    losses = out["losses"]
    print(
        "BLIP-mini dummy run | "
        f"images={tuple(images.shape)} | "
        f"text={tuple(input_ids.shape)} | "
        f"image_tokens={tuple(out['image_tokens'].shape)} | "
        f"ITC={losses['itc'].item():.4f}, "
        f"ITM={losses['itm'].item():.4f}, "
        f"LM={losses['lm'].item():.4f}, "
        f"total={losses['total'].item():.4f}"
    )

    print(f"sample text: {texts[0]}")


if __name__ == "__main__":
    main()
