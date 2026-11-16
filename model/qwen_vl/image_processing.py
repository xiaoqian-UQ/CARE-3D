import math
import os

import torch
from PIL import Image


DEFAULT_QWEN_IMAGE_EDGE = 336
# Qwen-VL treats shortest_edge/longest_edge as pixel budgets rather than
# literal edge lengths. 131072 produces a 24 x 24 patch grid for a square
# 336 x 336 image with the Qwen2.5/Qwen3 processors.
DEFAULT_QWEN_IMAGE_PIXELS = 131072


def _prepare_square_image(image, edge, pad, image_mean):
    image = image.convert("RGB")
    if not pad:
        return image.resize((edge, edge))

    width, height = image.size
    background = tuple(int(value * 255) for value in image_mean)
    canvas = Image.new("RGB", (max(width, height), max(width, height)), background)
    canvas.paste(image, ((canvas.width - width) // 2, (canvas.height - height) // 2))
    return canvas.resize((edge, edge))


def process_images_qwen_vl(images, image_processor, image_aspect_ratio="resize"):
    """
    Convert PIL images to Qwen-VL visual token tensors.

    Qwen2.5-VL and Qwen3-VL processors both return flattened patch
    `pixel_values`. The SceneCOT vision wrapper currently transports only the
    flattened tensor, so use a fixed square grid that can be reconstructed
    exactly before calling the vision tower.
    """
    edge = int(os.environ.get("SCENECOT_QWEN_IMAGE_EDGE", DEFAULT_QWEN_IMAGE_EDGE))
    pixel_budget = int(os.environ.get("SCENECOT_QWEN_IMAGE_PIXELS", DEFAULT_QWEN_IMAGE_PIXELS))
    if edge <= 0 or pixel_budget <= 0:
        raise ValueError("SCENECOT_QWEN_IMAGE_EDGE and SCENECOT_QWEN_IMAGE_PIXELS must be positive")

    new_imgs = []
    image_mean = getattr(image_processor, "image_mean", (0.0, 0.0, 0.0))
    pad = image_aspect_ratio == "pad"
    if image_aspect_ratio not in ("pad", "resize"):
        raise ValueError(f"Unsupported Qwen-VL image_aspect_ratio: {image_aspect_ratio}")

    for image in images:
        image = _prepare_square_image(image, edge=edge, pad=pad, image_mean=image_mean)
        out = image_processor(
            images=image,
            return_tensors="pt",
            size={"shortest_edge": pixel_budget, "longest_edge": pixel_budget},
        )
        pixel_values = out["pixel_values"]
        patch_count = pixel_values.shape[-2]
        patch_side = math.isqrt(patch_count)
        if patch_side * patch_side != patch_count:
            raise ValueError(
                "Qwen-VL preprocessing must produce a square patch grid, "
                f"but received {patch_count} flattened patches"
            )
        new_imgs.append(pixel_values)

    if all(x.shape == new_imgs[0].shape for x in new_imgs):
        return torch.stack(new_imgs, dim=0)
    return new_imgs


# Backward-compatible alias for older local imports/config comments.
process_images_qwen_2_5_vl = process_images_qwen_vl
