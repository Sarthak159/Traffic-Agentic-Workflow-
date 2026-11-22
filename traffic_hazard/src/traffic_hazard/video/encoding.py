"""Frame preprocessing: aspect-preserving resize, canonical pixel hashing, JPEG and contact sheets.

Changing any behavior here must bump `ENCODING_VERSION`, because it changes frame hashes and window
IDs.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from ..hashing import sha256_hex

ENCODING_VERSION = "rgb24-bilinear-v1"
JPEG_QUALITY = 90


@dataclass(frozen=True)
class EncodedFrame:
    """A resized frame and the SHA-256 of its canonical pixel buffer."""

    image: np.ndarray
    content_sha256: str


def resize_to_width(image: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale to at most `max_width` pixels wide, preserving aspect ratio; never upscale."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    new_height = max(1, round(height * max_width / width))
    resized = Image.fromarray(image, mode="RGB").resize((max_width, new_height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def pixel_sha256(image: np.ndarray) -> str:
    """Hash decoded RGB pixels with their shape, independent of container or file encoding."""
    height, width = image.shape[:2]
    header = f"{ENCODING_VERSION}:{width}x{height}:rgb24:".encode()
    return sha256_hex(header + np.ascontiguousarray(image, dtype=np.uint8).tobytes())


def encode_frame(image: np.ndarray, max_width: int) -> EncodedFrame:
    resized = resize_to_width(image, max_width)
    return EncodedFrame(image=resized, content_sha256=pixel_sha256(resized))


def jpeg_bytes(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def contact_sheet(
    images: Sequence[np.ndarray], labels: Sequence[str], columns: int = 5, tile_width: int = 240
) -> Image.Image:
    """Grid of frames with a text label (frame ID / source time) under each tile, for manual
    inspection.
    """
    if not images or len(images) != len(labels):
        raise ValueError("contact_sheet needs one label per image and at least one image")
    tiles = [Image.fromarray(resize_to_width(img, tile_width), mode="RGB") for img in images]
    tile_h = max(tile.height for tile in tiles)
    label_h = 16
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_h + label_h)), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for i, (tile, label) in enumerate(zip(tiles, labels, strict=True)):
        x, y = (i % columns) * tile_width, (i // columns) * (tile_h + label_h)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + tile_h + 2), label, fill=(255, 255, 255))
    return sheet
