from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _base_normalization() -> list:
    return [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(transpose_mask=True),
    ]


def build_transforms(preset: str, image_size: int):
    preset = preset.lower()
    dropout_min = max(8, int(round(image_size * 0.05)))
    dropout_max = max(dropout_min + 1, int(round(image_size * 0.12)))

    if preset == "weak":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(0.0, 0.03),
                    rotate=(-8, 8),
                    shear=(-3, 3),
                    p=0.5,
                ),
                *_base_normalization(),
            ]
        )

    if preset == "strong":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.7, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.85, 1.15),
                    translate_percent=(0.0, 0.08),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.15,
                            hue=0.05,
                            p=1.0,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2,
                            contrast_limit=0.2,
                            p=1.0,
                        ),
                    ],
                    p=0.6,
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 6),
                    hole_height_range=(dropout_min, dropout_max),
                    hole_width_range=(dropout_min, dropout_max),
                    fill=0,
                    fill_mask=0,
                    p=0.35,
                ),
                *_base_normalization(),
            ]
        )

    if preset == "val":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                *_base_normalization(),
            ]
        )

    raise ValueError(f"Unsupported augmentation preset: {preset}")
