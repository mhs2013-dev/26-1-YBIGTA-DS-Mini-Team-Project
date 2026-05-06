import torch


class SimpleTokenizer:
    """
    외부 tokenizer 없이 CLIP 흐름을 확인하기 위한 작은 문자 단위 tokenizer.
    실제 CLIP은 BPE tokenizer를 쓴다.
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, vocab_size: int, context_length: int) -> None:
        if vocab_size <= 8:
            raise ValueError("vocab_size should be bigger than 8")
        self.vocab_size = vocab_size
        self.context_length = context_length

    def encode(self, texts: list[str]) -> torch.Tensor:
        rows: list[list[int]] = []

        for text in texts:
            ids = [self.bos_token_id]
            for ch in text.lower()[: self.context_length - 2]:
                ids.append(3 + (ord(ch) % (self.vocab_size - 3)))
            ids.append(self.eos_token_id)

            pad_count = self.context_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_count
            rows.append(ids[: self.context_length])

        # token_ids: [batch, context_length]
        return torch.tensor(rows, dtype=torch.long)
