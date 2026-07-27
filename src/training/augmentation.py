import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_phone_augmentation(training: bool = True) -> T.Compose:
    """
    Augmentation pipeline for phone/horizontal view images.

    During training, applies random crop, flip, and color jitter
    to prevent the model from learning shortcuts (color histograms,
    fixed framing). During eval, applies only resize + normalize.
    """
    if training:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_satellite_augmentation(training: bool = True) -> T.Compose:
    """
    Augmentation pipeline for satellite/top-down view images.

    Slightly more aggressive color jitter than phone, since satellite
    imagery has more consistent lighting (no shadows from buildings).
    Random crop scale starts smaller (0.5) to handle varying resolutions.
    """
    if training:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
