from __future__ import annotations

import warnings

import segmentation_models_pytorch as smp


SUPPORTED_MODELS = (
    "deeplabv3plus_resnet50",
    "segformer_b0",
    "upernet_swin_tiny",
)


def _require_architecture(attr_name: str, model_name: str):
    if not hasattr(smp, attr_name):
        version = getattr(smp, "__version__", "unknown")
        raise ImportError(
            f"The installed segmentation_models_pytorch=={version} does not expose "
            f"`{attr_name}`, which is required for `{model_name}`. "
            "Upgrade segmentation-models-pytorch to a newer release that includes "
            "Segformer and UPerNet, then restart the notebook kernel."
        )
    return getattr(smp, attr_name)


def _build_with_fallback(model_cls, model_name: str, **kwargs):
    try:
        return model_cls(**kwargs)
    except Exception as exc:
        if kwargs.get("encoder_weights") is None:
            raise
        warnings.warn(
            f"Falling back to encoder_weights=None for `{model_name}` because pretrained "
            f"weights could not be loaded: {exc}",
            RuntimeWarning,
        )
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["encoder_weights"] = None
        return model_cls(**fallback_kwargs)


def build_model(model_name: str, num_classes: int, image_size: int | None = None):
    model_name = model_name.lower()

    if model_name == "deeplabv3plus_resnet50":
        model_cls = _require_architecture("DeepLabV3Plus", model_name)
        return _build_with_fallback(
            model_cls,
            model_name=model_name,
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

    if model_name == "segformer_b0":
        model_cls = _require_architecture("Segformer", model_name)
        return _build_with_fallback(
            model_cls,
            model_name=model_name,
            encoder_name="mit_b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

    if model_name == "upernet_swin_tiny":
        if image_size is None:
            warnings.warn(
                "`image_size` was not provided for `upernet_swin_tiny`; defaulting to 224 "
                "for model construction. Pass 192 or 384 explicitly during experiments.",
                RuntimeWarning,
            )
            image_size = 224
        model_cls = _require_architecture("UPerNet", model_name)
        return _build_with_fallback(
            model_cls,
            model_name=model_name,
            encoder_name="tu-swin_tiny_patch4_window7_224",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            img_size=image_size,
        )

    raise ValueError(f"Unsupported model_name={model_name!r}. Supported: {SUPPORTED_MODELS}")
