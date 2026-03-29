from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


def _load_rgb_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.uint8)


def _resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    img_resized = np.array(
        Image.fromarray(image).resize((width, height), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    mask_resized = np.array(
        Image.fromarray(mask).resize((width, height), resample=Image.NEAREST),
        dtype=np.uint8,
    )
    return img_resized, mask_resized


class _ManifestBackedTransform(BaseTransform):
    def __init__(self, manifest_path: str, prob: float = 0.5):
        self.manifest_path = str(manifest_path)
        self.prob = float(prob)
        self._manifest_df: pd.DataFrame | None = None

    def _ensure_manifest(self) -> pd.DataFrame:
        if self._manifest_df is None:
            self._manifest_df = pd.read_csv(self.manifest_path)
        return self._manifest_df

    def _sample_extra_pair(self) -> tuple[np.ndarray, np.ndarray]:
        manifest = self._ensure_manifest()
        row = manifest.sample(n=1).iloc[0]
        image = _load_rgb_image(row["image_path"])
        mask = _load_mask(row["mask_path"])
        return image, mask


@TRANSFORMS.register_module()
class FashionSegCutOut(BaseTransform):
    def __init__(
        self,
        prob: float = 0.3,
        num_holes: int = 2,
        min_ratio: float = 0.05,
        max_ratio: float = 0.12,
        fill_value: int = 0,
    ):
        self.prob = float(prob)
        self.num_holes = int(num_holes)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.fill_value = int(fill_value)

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        image = results["img"].copy()
        height, width = image.shape[:2]

        for _ in range(self.num_holes):
            hole_h = max(1, int(height * random.uniform(self.min_ratio, self.max_ratio)))
            hole_w = max(1, int(width * random.uniform(self.min_ratio, self.max_ratio)))
            y0 = random.randint(0, max(0, height - hole_h))
            x0 = random.randint(0, max(0, width - hole_w))
            image[y0 : y0 + hole_h, x0 : x0 + hole_w] = self.fill_value

        results["img"] = image
        results["img_shape"] = image.shape[:2]
        return results


@TRANSFORMS.register_module()
class FashionSegMixUp(_ManifestBackedTransform):
    """Segmentation-safe CutMix-style augmentation.

    The branch is named mix for the experiment plan, but the label mixing itself is
    implemented as a hard spatial replacement so the mask remains valid.
    """

    def __init__(
        self,
        manifest_path: str,
        prob: float = 0.5,
        min_ratio: float = 0.3,
        max_ratio: float = 0.7,
    ):
        super().__init__(manifest_path=manifest_path, prob=prob)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        image = results["img"].copy()
        mask = results["gt_seg_map"].copy()
        height, width = image.shape[:2]

        other_image, other_mask = self._sample_extra_pair()
        other_image, other_mask = _resize_image_and_mask(other_image, other_mask, (width, height))

        mix_h = max(1, int(height * random.uniform(self.min_ratio, self.max_ratio)))
        mix_w = max(1, int(width * random.uniform(self.min_ratio, self.max_ratio)))
        y0 = random.randint(0, max(0, height - mix_h))
        x0 = random.randint(0, max(0, width - mix_w))

        image[y0 : y0 + mix_h, x0 : x0 + mix_w] = other_image[y0 : y0 + mix_h, x0 : x0 + mix_w]
        mask[y0 : y0 + mix_h, x0 : x0 + mix_w] = other_mask[y0 : y0 + mix_h, x0 : x0 + mix_w]

        results["img"] = image
        results["gt_seg_map"] = mask
        results["img_shape"] = image.shape[:2]
        return results


@TRANSFORMS.register_module()
class FashionSegMosaic(_ManifestBackedTransform):
    def __init__(self, manifest_path: str, prob: float = 0.5):
        super().__init__(manifest_path=manifest_path, prob=prob)

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        image = results["img"]
        mask = results["gt_seg_map"]
        height, width = image.shape[:2]

        extra_pairs = [self._sample_extra_pair() for _ in range(3)]
        pairs = [(image, mask)] + extra_pairs

        split_x = int(width * random.uniform(0.35, 0.65))
        split_y = int(height * random.uniform(0.35, 0.65))

        tile_sizes = [
            (split_x, split_y),
            (width - split_x, split_y),
            (split_x, height - split_y),
            (width - split_x, height - split_y),
        ]
        tile_offsets = [
            (0, 0),
            (split_x, 0),
            (0, split_y),
            (split_x, split_y),
        ]

        mosaic_image = np.zeros((height, width, 3), dtype=np.uint8)
        mosaic_mask = np.zeros((height, width), dtype=np.uint8)

        for (src_image, src_mask), (tile_w, tile_h), (offset_x, offset_y) in zip(
            pairs,
            tile_sizes,
            tile_offsets,
        ):
            resized_image, resized_mask = _resize_image_and_mask(src_image, src_mask, (tile_w, tile_h))
            mosaic_image[offset_y : offset_y + tile_h, offset_x : offset_x + tile_w] = resized_image
            mosaic_mask[offset_y : offset_y + tile_h, offset_x : offset_x + tile_w] = resized_mask

        results["img"] = mosaic_image
        results["gt_seg_map"] = mosaic_mask
        results["img_shape"] = mosaic_image.shape[:2]
        return results
