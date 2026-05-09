"""
Evaluation metrics from Fong & Vedaldi (2017), Sections 5.2, 5.3, and 5.6.

§5.2  Deletion region representativeness
      – Simplified binary masks derived from the learned mask are used to
        perturb a set of images; normalised softmax suppression is plotted vs
        threshold α ∈ [0, 0.95].

§5.3  Minimality of deletions
      – Threshold the heatmap, fit a tight bounding box, blur the image inside
        that box, and report the smallest box that achieves a given suppression
        level (80 / 90 / 95 / 99 %).

§5.6  Localization error (Table 1)
      – Three thresholding strategies (value / energy / mean) are used to fit a
        bounding box and the IoU with the ground-truth box is tested at 0.5.

All functions are stateless and work on CPU or CUDA tensors.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from perturbation import (
    apply_blur_perturbation,
    apply_constant_perturbation,
    apply_noise_perturbation,
    GaussianBlur,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward_score(model, img, target_class):
    """Return softmax probability for target_class without tracking gradients."""
    with torch.no_grad():
        out = model(img)
        logits = (
            out["logits"] if isinstance(out, dict)
            else (out[0] if isinstance(out, (tuple, list)) else out)
        )
        return F.softmax(logits, dim=1)[0, target_class].item()


def baseline_scores(model, img, target_class, perturb_type="blur", blur_sigma=10):
    """
    Return (p0, pb):
        p0 = softmax score on the original image.
        pb = softmax score on the fully-perturbed image (all regions deleted).

    These are the denominators for the normalised-score formula (footnote 4).
    """
    device = img.device
    blur_module = GaussianBlur(sigma=blur_sigma).to(device)

    p0 = _forward_score(model, img, target_class)

    zero_mask = torch.zeros(1, 1, img.shape[2], img.shape[3], device=device)
    if perturb_type == "blur":
        fully_pert = apply_blur_perturbation(img, zero_mask, blur_module=blur_module)
    elif perturb_type == "constant":
        fully_pert = apply_constant_perturbation(img, zero_mask)
    else:
        fully_pert = apply_noise_perturbation(img, zero_mask)

    pb = _forward_score(model, fully_pert, target_class)
    return p0, pb


def normalized_score(p, p0, pb):
    """
    Normalised softmax probability (paper footnote 4, §5.2):

        p' = (p0 - p) / (p0 - pb)

    A value of 1.0 means complete suppression (score dropped to pb).
    A value of 0.0 means no suppression at all.

    Args:
        p:  Score of the masked image.
        p0: Score of the original image.
        pb: Score of the fully-perturbed image.
    """
    denom = p0 - pb
    if abs(denom) < 1e-8:
        return 0.0
    return float((p0 - p) / denom)


# ---------------------------------------------------------------------------
# Mask manipulation utilities
# ---------------------------------------------------------------------------

def smooth_mask(mask_np, smooth_sigma=10):
    """
    Further Gaussian-blur a mask array (§5.2: 'simplify our masks by further
    blurring them').  mask_np should be a 2-D float array in [0, 1].
    """
    if smooth_sigma <= 0:
        return mask_np.astype(np.float32)
    ks = max(3, int(2 * np.ceil(3 * smooth_sigma) + 1))
    if ks % 2 == 0:
        ks += 1
    return cv2.GaussianBlur(mask_np.astype(np.float32), (ks, ks), smooth_sigma)


def threshold_to_deletion_binary(mask_np, alpha):
    """
    Convert a continuous mask to a binary deletion map (§5.2, §5.3).

    Saliency = 1 - mask  (deleted regions are salient).
    Returns binary array: 1 = salient/deleted, 0 = kept.
    """
    saliency = 1.0 - mask_np
    return (saliency >= alpha).astype(np.float32)


def mask_to_bbox(binary_mask):
    """
    Fit the tightest axis-aligned bounding box around non-zero pixels (§5.3, §5.6).

    Returns (y1, x1, y2, x2) in pixel coordinates, or None if mask is empty.
    """
    rows = np.any(binary_mask > 0, axis=1)
    cols = np.any(binary_mask > 0, axis=0)
    if not rows.any():
        return None
    y1, y2 = int(np.where(rows)[0][[0, -1]].tolist()[0]), int(np.where(rows)[0][[0, -1]].tolist()[1])
    x1, x2 = int(np.where(cols)[0][[0, -1]].tolist()[0]), int(np.where(cols)[0][[0, -1]].tolist()[1])
    return (y1, x1, y2, x2)


def bbox_iou(pred, gt):
    """
    Intersection-over-union of two bounding boxes in (y1, x1, y2, x2) format.
    Used for localization error in §5.6 (IoU threshold = 0.5).
    """
    if pred is None or gt is None:
        return 0.0
    py1, px1, py2, px2 = pred
    gy1, gx1, gy2, gx2 = gt
    iy1 = max(py1, gy1)
    ix1 = max(px1, gx1)
    iy2 = min(py2, gy2)
    ix2 = min(px2, gx2)
    inter = max(0, iy2 - iy1 + 1) * max(0, ix2 - ix1 + 1)
    pred_area = (py2 - py1 + 1) * (px2 - px1 + 1)
    gt_area = (gy2 - gy1 + 1) * (gx2 - gx1 + 1)
    union = pred_area + gt_area - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# §5.2  Deletion region representativeness
# ---------------------------------------------------------------------------

def suppression_curve(
    model,
    img,
    mask_np,
    target_class,
    p0,
    pb,
    alphas=None,
    perturb_types=None,
    blur_sigma=10,
    extra_smooth_sigma=10,
):
    """
    §5.2: For each α in `alphas`, threshold the (further-smoothed) learned mask
    to produce a binary deletion map, apply each perturbation type, and record
    the normalised suppression score.

    This reproduces Figure 7 (bottom) of the paper.

    Args:
        model:              Classifier.
        img:                Preprocessed image tensor (1, 3, H, W).
        mask_np:            Learned mask as a 2-D numpy array in [0, 1].
                            m=1 = keep, m=0 = deleted/salient.
        target_class:       Class index to evaluate.
        p0:                 Original score (from baseline_scores).
        pb:                 Fully-perturbed score (from baseline_scores).
        alphas:             Thresholds to sweep. Default: 0, 0.05, …, 0.95.
        perturb_types:      List of perturbation types. Default: all three.
        blur_sigma:         σ for blur perturbation operator.
        extra_smooth_sigma: Additional smoothing applied before thresholding
                            (paper: 'further blurring').

    Returns:
        alphas:  numpy array of threshold values.
        results: dict  {perturb_type: list of normalised suppression scores}.
    """
    if alphas is None:
        alphas = np.arange(0.0, 1.0, 0.05)
    if perturb_types is None:
        perturb_types = ["blur", "constant", "noise"]

    device = img.device
    blur_module = GaussianBlur(sigma=blur_sigma).to(device)

    # Further smooth, then normalise to [0, 1]
    smoothed = smooth_mask(mask_np, smooth_sigma=extra_smooth_sigma)
    lo, hi = smoothed.min(), smoothed.max()
    smoothed = (smoothed - lo) / (hi - lo + 1e-8)

    results = {pt: [] for pt in perturb_types}

    for alpha in alphas:
        # Binary deletion map: 1 = deleted region, 0 = kept
        deletion = threshold_to_deletion_binary(smoothed, alpha)
        # Convert to keep-mask for perturbation operators (m=1 keep, m=0 perturb)
        keep_mask = torch.tensor(
            1.0 - deletion, dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0)

        for pt in perturb_types:
            if pt == "blur":
                pert = apply_blur_perturbation(img, keep_mask, blur_module=blur_module)
            elif pt == "constant":
                pert = apply_constant_perturbation(img, keep_mask)
            else:
                pert = apply_noise_perturbation(img, keep_mask)

            p = _forward_score(model, pert, target_class)
            results[pt].append(normalized_score(p, p0, pb))

    return alphas, results


# ---------------------------------------------------------------------------
# §5.3  Minimality of deletions
# ---------------------------------------------------------------------------

def minimality_curve(
    model,
    img,
    heatmap_np,
    target_class,
    p0,
    pb,
    thresholds=None,
    blur_sigma=10,
):
    """
    §5.3: For each threshold h, fit the tightest bounding box around the
    thresholded heatmap, blur the image inside that box, and record the
    normalised suppression score and box size.

    This reproduces Figure 8 of the paper.

    Args:
        heatmap_np:  Saliency heatmap (= 1 - mask) as a 2-D numpy array.
                     Higher values indicate more salient/deleted regions.
        thresholds:  Values of h to sweep. Default: 0.0, 0.1, …, 1.0.

    Returns:
        thresholds:        numpy array.
        norm_scores:       list of normalised suppression values per threshold.
        bbox_area_fracs:   list of bounding-box area fractions (0–1) per threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.1, 0.1)

    device = img.device
    H, W = img.shape[2], img.shape[3]
    blur_module = GaussianBlur(sigma=blur_sigma).to(device)

    h_norm = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

    norm_scores = []
    bbox_area_fracs = []

    for h in thresholds:
        binary = (h_norm >= h).astype(np.float32)
        bbox = mask_to_bbox(binary)

        if bbox is None:
            norm_scores.append(0.0)
            bbox_area_fracs.append(0.0)
            continue

        y1, x1, y2, x2 = bbox
        area_frac = float((y2 - y1 + 1) * (x2 - x1 + 1)) / (H * W)
        bbox_area_fracs.append(area_frac)

        # Blur inside bounding box (keep the rest)
        box_mask = torch.ones(1, 1, H, W, device=device)
        box_mask[:, :, y1:y2 + 1, x1:x2 + 1] = 0.0
        pert = apply_blur_perturbation(img, box_mask, blur_module=blur_module)

        p = _forward_score(model, pert, target_class)
        norm_scores.append(normalized_score(p, p0, pb))

    return thresholds, norm_scores, bbox_area_fracs


def minimality_summary(thresholds, norm_scores, bbox_area_fracs,
                       suppression_levels=(0.80, 0.90, 0.95, 0.99)):
    """
    §5.3 (Figure 8): For each suppression level, return the smallest bounding-
    box area fraction (as a fraction of image area) that achieves it.

    Returns:
        dict  {suppression_level: min_bbox_area_frac or None if not reached}
    """
    result = {}
    for level in suppression_levels:
        min_area = None
        for score, area in zip(norm_scores, bbox_area_fracs):
            if score >= level and (min_area is None or area < min_area):
                min_area = area
        result[level] = min_area
    return result


# ---------------------------------------------------------------------------
# §5.6  Localization error (Table 1)
# ---------------------------------------------------------------------------

def localization_error_single(
    heatmap_np,
    gt_bbox,
    method="value",
    alphas=None,
    iou_threshold=0.5,
):
    """
    §5.6: Compute localization error for one image using one thresholding
    strategy.  Returns (best_alpha, is_error) where is_error ∈ {0, 1}.

    Args:
        heatmap_np:    2-D saliency array.  Higher = more salient.
        gt_bbox:       Ground-truth bounding box (y1, x1, y2, x2).
        method:        "value"  – threshold by intensity α.
                       "energy" – threshold so top-α fraction of energy is kept.
                       "mean"   – threshold by τ = α·mean_intensity.
        alphas:        Search grid. Defaults match paper Table 1.
        iou_threshold: IoU ≥ this → correct localisation. Paper uses 0.5.

    Returns:
        best_alpha: α that gives lowest error on this image.
        errors:     list of per-alpha error values (0 or 1).
    """
    if alphas is None:
        if method == "mean":
            alphas = np.arange(0.0, 10.5, 0.5)
        else:
            alphas = np.arange(0.0, 1.0, 0.05)

    h_norm = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    mean_intensity = float(h_norm.mean())
    total_energy = float(h_norm.sum())

    errors = []
    for alpha in alphas:
        if method == "value":
            binary = (h_norm >= float(alpha)).astype(np.float32)

        elif method == "energy":
            # Keep pixels whose cumulative energy (sorted descending) ≤ α * total
            flat = h_norm.ravel()
            order = np.argsort(flat)[::-1]
            cumsum = np.cumsum(flat[order])
            keep_n = int(np.searchsorted(cumsum, float(alpha) * total_energy)) + 1
            cutoff = flat[order[min(keep_n - 1, len(order) - 1)]]
            binary = (h_norm >= cutoff).astype(np.float32)

        elif method == "mean":
            binary = (h_norm >= float(alpha) * mean_intensity).astype(np.float32)

        else:
            binary = (h_norm >= float(alpha)).astype(np.float32)

        pred_bbox = mask_to_bbox(binary)
        iou = bbox_iou(pred_bbox, gt_bbox)
        errors.append(0 if iou >= iou_threshold else 1)

    best_idx = int(np.argmin(errors))
    return float(alphas[best_idx]), errors


def aggregate_localization_error(per_image_errors):
    """
    §5.6 (Table 1): Given a list of per-image error lists (one list per image,
    each of length |alphas|), return the overall error rate at the optimal α
    found by choosing the best α separately for each image (optimistic bound,
    matches paper's 'optimal α on a heldout set' protocol).

    Args:
        per_image_errors: list of lists, shape (N_images, N_alphas).

    Returns:
        (best_alpha_index, error_rate_percent)
    """
    arr = np.array(per_image_errors, dtype=np.float32)  # (N, A)
    mean_errors = arr.mean(axis=0)                        # (A,)
    best_idx = int(np.argmin(mean_errors))
    return best_idx, float(mean_errors[best_idx]) * 100.0


# ---------------------------------------------------------------------------
# §5.5  Adversarial examples – adversarial mask comparison
# ---------------------------------------------------------------------------

def generate_adversarial_fgsm(model, img, target_class, epsilon=8.0 / 255.0):
    """
    One-step iterative FGSM adversarial example (paper §5.5, ε=8/255).
    Uses the sign of the gradient of the cross-entropy loss w.r.t. the input.

    Returns: adversarial image tensor (1, 3, H, W), same device as img.
    """
    img_adv = img.clone().detach().requires_grad_(True)
    out = model(img_adv)
    logits = (
        out["logits"] if isinstance(out, dict)
        else (out[0] if isinstance(out, (tuple, list)) else out)
    )
    loss = F.cross_entropy(logits, torch.tensor([target_class], device=img.device))
    loss.backward()
    perturbation = epsilon * img_adv.grad.sign()
    return (img + perturbation).detach()
