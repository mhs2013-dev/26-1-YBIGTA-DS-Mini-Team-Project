import torch

from .config import CLIPConfig
from .modules.tokenizer import SimpleTokenizer


def make_dummy_batch(config: CLIPConfig) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    raw_texts = [
        "a photo of a dog",
        "a photo of a bicycle",
        "a photo of a coffee cup",
        "a photo of a city street",
    ]
    tokenizer = SimpleTokenizer(config.vocab_size, config.max_text_len)

    # images: [batch, channels, height, width]
    images = torch.randn(len(raw_texts), 3, config.image_size, config.image_size)

    # input_ids: [batch, text_length]
    input_ids = tokenizer.encode(raw_texts)
    return images, input_ids, raw_texts
