import torch
import torch.nn.functional as F


def clip_contrastive_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor,
) -> torch.Tensor:
    # logits_per_image: [batch, batch]
    # logits_per_text: [batch, batch]
    batch_size = logits_per_image.size(0)
    targets = torch.arange(batch_size, device=logits_per_image.device)

    image_loss = F.cross_entropy(logits_per_image, targets)
    text_loss = F.cross_entropy(logits_per_text, targets)
    return (image_loss + text_loss) / 2
