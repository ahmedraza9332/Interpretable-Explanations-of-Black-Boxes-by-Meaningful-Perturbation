"""
Mask optimization: the "deletion game" from Fong & Vedaldi (2017).

Learns a mask that, when used to perturb the image, minimizes the
classifier's score for the target class. Implements Equation 4 of the paper:

    min_{m in [0,1]} λ1·||1-m||_1  +  λ2·Σ||∇m||^β_β
                     + E_τ[f_c(Φ(x0(·-τ), m))]

Mask convention: m=1 → keep original pixel, m=0 → apply perturbation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from perturbation import (
    apply_blur_perturbation,
    apply_constant_perturbation,
    apply_noise_perturbation,
    GaussianBlur,
    upsample_mask,
)


def tv_norm(mask, beta=3):
    """Total variation regularisation (paper §4.3, Eq. 4, λ2 term).
    Computes Σ_u ||∇m(u)||^β_β over horizontal and vertical differences.
    beta=3 matches the paper default.
    """
    diff_h = mask[:, :, 1:, :] - mask[:, :, :-1, :]
    diff_w = mask[:, :, :, 1:] - mask[:, :, :, :-1]
    return diff_h.abs().pow(beta).mean() + diff_w.abs().pow(beta).mean()


def _get_score(model, img, target_class):
    """Return softmax probability for target_class (no grad)."""
    with torch.no_grad():
        out = model(img)
        logits = (
            out["logits"] if isinstance(out, dict)
            else (out[0] if isinstance(out, (tuple, list)) else out)
        )
        return F.softmax(logits, dim=1)[0, target_class].item()


def _init_circular_mask(
    model,
    img,
    target_class,
    mask_res,
    img_size,
    perturb_type,
    blur_module,
    suppression_threshold=0.99,
):
    """
    Paper §4.3 (Implementation details):
    'Initialize the mask as the smallest centered circular mask that suppresses
    the score of the original image by 99% when compared to that of the fully
    perturbed image.'

    Mask convention: 0 inside the circle (deleted), 1 outside (kept).
    We binary-search over the radius in low-resolution mask space.

    Returns a (1, 1, mask_res[0], mask_res[1]) float tensor in [0, 1].
    """
    device = img.device
    h, w = mask_res

    # Baseline scores
    p0 = _get_score(model, img, target_class)

    zero_mask = torch.zeros(1, 1, *img_size, device=device)
    if perturb_type == "blur":
        fully_pert = apply_blur_perturbation(img, zero_mask, blur_module=blur_module)
    elif perturb_type == "constant":
        fully_pert = apply_constant_perturbation(img, zero_mask)
    else:
        fully_pert = apply_noise_perturbation(img, zero_mask)
    pb = _get_score(model, fully_pert, target_class)

    # Target: p_init ≤ p0 - suppression_threshold * (p0 - pb)
    # i.e. normalised suppression ≥ suppression_threshold
    target_score = p0 - suppression_threshold * (p0 - pb)

    # Pre-compute distance-from-centre grid in low-res space
    ys = torch.arange(h, device=device).float()
    xs = torch.arange(w, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dist = ((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2).sqrt()

    max_r = int((min(h, w) / 2.0))
    best_r = max_r  # fallback: full mask

    for r in range(1, max_r + 1):
        # 0 inside circle (deleted), 1 outside (kept)
        circle = (dist > r).float().unsqueeze(0).unsqueeze(0)
        mask_up = upsample_mask(circle, img_size, smooth_sigma=0)

        if perturb_type == "blur":
            pert = apply_blur_perturbation(img, mask_up, blur_module=blur_module)
        elif perturb_type == "constant":
            pert = apply_constant_perturbation(img, mask_up)
        else:
            pert = apply_noise_perturbation(img, mask_up)

        score = _get_score(model, pert, target_class)
        if score <= target_score:
            best_r = r
            break

    init = (dist > best_r).float().unsqueeze(0).unsqueeze(0)
    return init


def optimize_mask(
    model,
    img,
    target_class=None,
    mask_res=(28, 28),
    img_size=(224, 224),
    n_iter=300,
    lr=0.1,
    l1_coeff=1e-4,
    tv_coeff=1e-2,
    tv_beta=3,
    blur_sigma=10,
    mask_smooth_sigma=5,
    perturb_type="blur",
    use_jitter=True,
    jitter_max=4,
    circular_init=True,
    verbose=True,
):
    """
    Learn deletion mask via gradient descent (Equation 4, paper §4.3).

    Args:
        model:            Classifier (callable, returns logits or dict/tuple).
        img:              Preprocessed image tensor (1, 3, H, W).
        target_class:     Class index to suppress. None → top predicted class.
        mask_res:         Low-res mask size (h, w). Paper uses 28×28 for GoogLeNet.
        img_size:         (H, W) for upsampling. Default 224×224.
        n_iter:           Gradient steps. Paper default: 300.
        lr:               Adam learning rate (γ). Paper default: 0.1.
        l1_coeff:         λ1 for sparsity  ||1-m||_1. Paper default: 1e-4.
        tv_coeff:         λ2 for TV regularisation. Paper default: 1e-2.
        tv_beta:          Exponent β for TV norm. Paper default: 3.
        blur_sigma:       σ0 for blur perturbation. Paper uses 10.
        mask_smooth_sigma: σm for smoothing upsampled mask. Paper uses 5.
        perturb_type:     "blur" | "constant" | "noise".
        use_jitter:       Random translation during optimisation (paper §4.3).
        jitter_max:       τ in paper. Shift drawn from [0, τ) uniform.
        circular_init:    If True, use the paper's circular-mask initialisation
                          (§4.3); otherwise start from all-ones.
        verbose:          Print progress every 50 iterations.

    Returns:
        mask:          Learned mask (1, 1, H, W), values in [0, 1].
        perturbed:     Final perturbed image.
        initial_score: Softmax score before optimisation.
        final_score:   Softmax score after optimisation.
        target_class:  Class index that was suppressed.
    """
    device = img.device
    blur_module = GaussianBlur(sigma=blur_sigma).to(device)

    # Determine target class and record initial score
    with torch.no_grad():
        out = model(img)
        logits = (
            out["logits"] if isinstance(out, dict)
            else (out[0] if isinstance(out, (tuple, list)) else out)
        )
        probs = F.softmax(logits, dim=1)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        initial_score = probs[0, target_class].item()

    # --- Mask initialisation ---
    if circular_init:
        if verbose:
            print("  Finding circular initialisation mask (paper §4.3)...")
        init_val = _init_circular_mask(
            model, img, target_class,
            mask_res, img_size, perturb_type, blur_module,
        )
    else:
        init_val = torch.ones(1, 1, mask_res[0], mask_res[1], device=device)

    mask_low = nn.Parameter(init_val.clone().to(device))
    optimizer = torch.optim.Adam([mask_low], lr=lr)

    for i in range(n_iter):
        optimizer.zero_grad()

        # Upsample and smooth low-res mask to image size
        mask_up = upsample_mask(mask_low, img_size, smooth_sigma=mask_smooth_sigma)

        # Stochastic jitter: draw shift from discrete uniform [0, τ)  (paper §4.3)
        if use_jitter and jitter_max > 0:
            dx = torch.randint(0, jitter_max, (1,), device=device).item()
            dy = torch.randint(0, jitter_max, (1,), device=device).item()
            if dx != 0 or dy != 0:
                mask_up = torch.roll(mask_up, (dy, dx), dims=(2, 3))

        # Apply perturbation operator Φ (paper §4.1)
        if perturb_type == "blur":
            perturbed = apply_blur_perturbation(img, mask_up, blur_module=blur_module)
        elif perturb_type == "constant":
            perturbed = apply_constant_perturbation(img, mask_up)
        elif perturb_type == "noise":
            perturbed = apply_noise_perturbation(img, mask_up)
        else:
            perturbed = apply_blur_perturbation(img, mask_up, blur_module=blur_module)

        # Forward pass
        out = model(perturbed)
        logits = (
            out["logits"] if isinstance(out, dict)
            else (out[0] if isinstance(out, (tuple, list)) else out)
        )
        score = F.softmax(logits, dim=1)[0, target_class]

        # Equation 4: f_c + λ1·||1-m||_1 + λ2·TV^β(m)
        l1_loss = (1 - mask_low).abs().mean()
        tv_loss = tv_norm(mask_up, beta=tv_beta)
        loss = score + l1_coeff * l1_loss + tv_coeff * tv_loss

        loss.backward()
        optimizer.step()

        # Project mask back to [0, 1]
        with torch.no_grad():
            mask_low.clamp_(0, 1)

        if verbose and (i + 1) % 50 == 0:
            print(f"  iter {i+1}/{n_iter}  score={score.item():.4f}  loss={loss.item():.4f}")

    # Final mask and score
    with torch.no_grad():
        mask_final = upsample_mask(mask_low, img_size, smooth_sigma=mask_smooth_sigma)
        perturbed_final = apply_blur_perturbation(img, mask_final, blur_module=blur_module)
        out = model(perturbed_final)
        logits = (
            out["logits"] if isinstance(out, dict)
            else (out[0] if isinstance(out, (tuple, list)) else out)
        )
        final_score = F.softmax(logits, dim=1)[0, target_class].item()

    return mask_final, perturbed_final, initial_score, final_score, target_class
