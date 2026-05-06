import torch

from clip_core import CLIPConfig, CLIPMini
from clip_core.data import make_dummy_batch


def main() -> None:
    torch.manual_seed(11)

    config = CLIPConfig()
    model = CLIPMini(config)
    model.eval()

    images, input_ids, raw_texts = make_dummy_batch(config)

    with torch.no_grad():
        out = model(images, input_ids)

    print(
        "CLIP-mini dummy run | "
        f"images={tuple(images.shape)} | "
        f"text={tuple(input_ids.shape)} | "
        f"logits={tuple(out['logits_per_image'].shape)} | "
        f"loss={out['loss'].item():.4f}"
    )
    print(f"sample text: {raw_texts[0]}")


if __name__ == "__main__":
    main()
