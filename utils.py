"""
Utilities: preprocessing, model loading, visualization.
"""

import numpy as np
import torch
from pathlib import Path

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_image(path, size=(224, 224)):
    """Load and preprocess image for model (ImageNet normalization)."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0
    if img.shape[:2] != size:
        from skimage.transform import resize
        img = resize(img, (*size, 3), anti_aliasing=True, preserve_range=False)
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    # Normalize
    img = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return img.astype(np.float32)


def preprocess_tensor(img_np, device):
    """Convert numpy image to model input tensor."""
    x = torch.from_numpy(img_np).unsqueeze(0).float().to(device)
    return x


def denormalize(img_tensor):
    """Convert normalized tensor back to [0,1] for display."""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = np.array(img_tensor)
    if img.ndim == 4:
        img = img[0]
    if img.shape[0] == 3:  # CHW
        img = np.transpose(img, (1, 2, 0))
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def load_model(name="vgg19", device=None):
    """Load pretrained classifier. Options: vgg19, googlenet, alexnet."""
    import torchvision.models as models
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = name.lower()
    if name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    elif name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    elif name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model: {name}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_class_label(idx):
    """Get human-readable ImageNet class name by index."""
    try:
        import urllib.request
        import json
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        cache = Path(__file__).parent / "_imagenet_labels.json"
        if not cache.exists():
            with urllib.request.urlopen(url, timeout=5) as r:
                labels = json.loads(r.read().decode())
            cache.write_text(json.dumps(labels))
        else:
            labels = json.loads(cache.read_text())
        return labels[idx] if idx < len(labels) else f"class_{idx}"
    except Exception:
        return f"class_{idx}"


def save_explanations(mask, original_img, perturbed_img, output_dir, prefix="out"):
    """Save mask, heatmap overlay, and perturbed image."""
    import cv2
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # mask: (1,1,H,W) or (H,W), values in [0,1]
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.array(mask)
    if m.ndim == 4:
        m = m[0, 0]
    elif m.ndim == 3:
        m = m[0]
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)

    # Saliency = 1 - mask (deletion region = high saliency)
    saliency = 1 - m
    heatmap = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    if isinstance(original_img, torch.Tensor):
        orig = denormalize(original_img)
    else:
        orig = np.array(original_img)
    if orig.ndim == 4:
        orig = orig[0]
    if orig.shape[0] == 3:
        orig = np.transpose(orig, (1, 2, 0))
    orig = np.clip(orig, 0, 1)
    if orig.max() <= 1:
        orig_uint8 = (orig * 255).astype(np.uint8)
    else:
        orig_uint8 = orig.astype(np.uint8)

    overlay = (0.5 * orig + 0.5 * heatmap)
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    if isinstance(perturbed_img, torch.Tensor):
        pert = denormalize(perturbed_img)
    else:
        pert = np.array(perturbed_img)
    if pert.ndim == 4:
        pert = pert[0]
    if pert.shape[0] == 3:
        pert = np.transpose(pert, (1, 2, 0))
    pert = np.clip(pert, 0, 1)
    pert_uint8 = (pert * 255).astype(np.uint8)

    cv2.imwrite(str(output_dir / f"{prefix}_original.png"), cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{prefix}_mask.png"), (m * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / f"{prefix}_heatmap.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{prefix}_perturbed.png"), cv2.cvtColor(pert_uint8, cv2.COLOR_RGB2BGR))


def save_mask_only(mask, path):
    """Save mask as grayscale image."""
    import cv2
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.array(mask)
    if m.ndim >= 3:
        m = m.squeeze()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    cv2.imwrite(str(path), (m * 255).astype(np.uint8))
