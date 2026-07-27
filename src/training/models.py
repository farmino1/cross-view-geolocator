import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    """
    Base encoder wrapper. Takes a backbone CNN and a linear projection
    head, outputs L2-normalized embeddings.
    """

    def __init__(self, backbone: nn.Module, backbone_dim: int, embed_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(backbone_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        projected = self.projection(features)
        return F.normalize(projected, dim=-1)

    def encode_batch(
        self, images: torch.Tensor, batch_size: int = 64
    ) -> torch.Tensor:
        """
        Encode a large batch of images in chunks.
        Useful for pre-computing satellite tile embeddings.

        Args:
            images: [N, 3, H, W] — images to encode
            batch_size: chunk size for memory efficiency

        Returns:
            [N, embed_dim] — L2-normalized embeddings
        """
        self.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                embeddings.append(self.forward(batch))
        return torch.cat(embeddings, dim=0)


def create_resnet50_encoder(embed_dim: int = 256, pretrained: bool = True) -> Encoder:
    """
    Create a ResNet-50 encoder with linear projection.

    Args:
        embed_dim: output embedding dimension
        pretrained: load ImageNet pretrained weights

    Returns:
        Encoder wrapping ResNet-50 backbone
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = models.resnet50(weights=weights)
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
    return Encoder(backbone, backbone_dim=2048, embed_dim=embed_dim)


def create_mobilenetv3_encoder(
    embed_dim: int = 256, pretrained: bool = True
) -> Encoder:
    """
    Create a MobileNetV3-Large encoder with linear projection.

    Args:
        embed_dim: output embedding dimension
        pretrained: load ImageNet pretrained weights

    Returns:
        Encoder wrapping MobileNetV3-Large backbone
    """
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    mobilenet = models.mobilenet_v3_large(weights=weights)
    # MobileNetV3-Large classifier: Sequential(Linear, Hardswish, Dropout, Linear, Hardswish)
    # We want everything except the classifier
    backbone = mobilenet.features
    return Encoder(backbone, backbone_dim=960, embed_dim=embed_dim)
