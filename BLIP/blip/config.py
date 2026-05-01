from dataclasses import dataclass

# 데코레이터로 정의된 BLIPConfig 클래스는 모델의 하이퍼파라미터와 설정을 저장하는 역할을 합니다.
@dataclass
class BLIPConfig:
    image_size: int = 64
    patch_size: int = 16
    vocab_size: int = 128
    max_text_len: int = 32
    embed_dim: int = 128
    vision_layers: int = 2
    text_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    temperature: float = 0.07
    # decorator로 정의된 property는 클래스의 속성처럼 접근할 수 있는 메서드입니다.
    # 예를 들어 config.num_patches로 접근하면 num_patches 메서드가 호출되어 계산된 값을 반환합니다.
    @property
    def num_patches(self) -> int:
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side
