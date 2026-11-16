from __future__ import annotations

import torch


def process_images_gemma3(images, image_processor):
    """Preprocess one or more PIL images for Gemma 3's SigLIP vision tower."""
    pixel_values = []
    for image in images:
        processed = image_processor(
            images=image.convert("RGB"),
            return_tensors="pt",
            do_pan_and_scan=False,
        )["pixel_values"]
        if processed.ndim != 4 or processed.shape[0] != 1:
            raise ValueError(
                "Gemma 3 preprocessing must produce exactly one image tensor, "
                f"got {tuple(processed.shape)}"
            )
        pixel_values.append(processed[0])

    if not pixel_values:
        raise ValueError("At least one image is required")
    if all(value.shape == pixel_values[0].shape for value in pixel_values):
        return torch.stack(pixel_values, dim=0)
    return pixel_values
