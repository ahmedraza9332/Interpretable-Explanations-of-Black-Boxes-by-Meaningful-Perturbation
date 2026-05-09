"""
Fetch a small set of sample images for the Meaningful Perturbation project.
Space-conscious: ~5 images, each 224x224, total < 500KB.
"""

import os
import sys
from pathlib import Path

# Create output directory
IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)
TARGET_SIZE = (224, 224)  # Required by VGG/GoogLeNet


def fetch_from_skimage():
    """Use scikit-image built-in images (no download, bundled with package)."""
    try:
        from skimage import data
        from skimage.transform import resize
        from PIL import Image
    except ImportError:
        print("scikit-image not installed. Run: pip install scikit-image")
        return []

    # Classic built-in images - bundled, no network download
    def _get(img_or_fn):
        return img_or_fn() if callable(img_or_fn) else img_or_fn

    samples = [
        ("astronaut", data.astronaut),
        ("coffee", data.coffee),
        ("cat", getattr(data, "cat", data.astronaut)),
    ]
    saved = []
    for name, img_src in samples:
        try:
            img = _get(img_src)
            if img.ndim == 2:
                from skimage.color import gray2rgb
                img = gray2rgb(img)
            img = resize(img, TARGET_SIZE, anti_aliasing=True, preserve_range=True).astype("uint8")
            path = IMAGES_DIR / f"{name}.jpg"
            Image.fromarray(img).save(path, "JPEG", quality=85)
            size_kb = path.stat().st_size / 1024
            saved.append((path.name, size_kb))
        except Exception as e:
            print(f"  Skipped {name}: {e}")
    return saved


def fetch_from_picsum(count=2):
    """Fetch a few small images from Lorem Picsum (no API key, 224x224)."""
    try:
        import urllib.request
        import numpy as np
        from PIL import Image
        from io import BytesIO
    except ImportError as e:
        print(f"Missing dependency for Picsum fetch: {e}")
        return []

    # Curated IDs: dog, cat, bird, car - good for ImageNet-style classification
    ids = [237, 1025, 219, 1074, 433]  # dog, beach, mountain, etc.
    saved = []
    for i in range(min(count, len(ids))):
        url = f"https://picsum.photos/id/{ids[i]}/{TARGET_SIZE[0]}/{TARGET_SIZE[1]}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                img = Image.open(BytesIO(resp.read())).convert("RGB")
            path = IMAGES_DIR / f"picsum_{ids[i]}.jpg"
            img.save(path, "JPEG", quality=85)
            size_kb = path.stat().st_size / 1024
            saved.append((path.name, size_kb))
        except Exception as e:
            print(f"  Skipped picsum id {ids[i]}: {e}")
    return saved


def main():
    print("Fetching sample images (space-conscious, < 1MB total)...")
    print(f"Output: {IMAGES_DIR}\n")

    all_saved = []

    # 1. Built-in scikit-image (no network)
    print("1. Loading built-in images from scikit-image (no download)...")
    all_saved.extend(fetch_from_skimage())

    # 2. A couple from Picsum
    print("2. Fetching 2 images from Lorem Picsum (224x224 each)...")
    all_saved.extend(fetch_from_picsum(count=2))

    if not all_saved:
        print("No images fetched. Install: pip install scikit-image Pillow")
        sys.exit(1)

    total_kb = sum(s[1] for s in all_saved)
    print(f"\nDone. Saved {len(all_saved)} images:")
    for name, size_kb in all_saved:
        print(f"  - {name}: {size_kb:.1f} KB")
    print(f"  Total: ~{total_kb:.0f} KB ({total_kb/1024:.2f} MB)")


if __name__ == "__main__":
    main()
