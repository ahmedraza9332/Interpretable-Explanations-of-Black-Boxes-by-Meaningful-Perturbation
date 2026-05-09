"""
Main entry point: Meaningful Perturbation explanations.

Implements the deletion game from Fong & Vedaldi (2017), Section 4.

Usage:
    python explain.py images/coffee.jpg
    python explain.py images/cat.jpg --model googlenet --iter 300 --out output/
"""

import argparse
from pathlib import Path

import torch

from utils import load_image, load_model, get_class_label, save_explanations, preprocess_tensor
from mask_optimizer import optimize_mask
from metrics import baseline_scores, normalized_score


def main():
    parser = argparse.ArgumentParser(
        description="Meaningful Perturbation — Interpretable Explanations of Black Boxes"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--model", type=str, default="googlenet",
        choices=["vgg19", "googlenet", "alexnet"],
        help="Classifier (paper uses googlenet, default).",
    )
    parser.add_argument("--iter", type=int, default=300,
                        help="Optimisation iterations (paper default: 300).")
    parser.add_argument("--out", type=str, default="output", help="Output directory.")
    parser.add_argument("--no-jitter", action="store_true",
                        help="Disable stochastic jitter (paper uses jitter by default).")
    parser.add_argument("--no-circular-init", action="store_true",
                        help="Disable circular-mask initialisation (paper §4.3).")
    parser.add_argument(
        "--perturb", type=str, default="blur",
        choices=["blur", "constant", "noise"],
        help="Perturbation type (paper default: blur).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: '{img_path}' not found.")
        return 1

    img_np = load_image(str(img_path))
    img = preprocess_tensor(img_np, device)

    print(f"Loading {args.model}...")
    model = load_model(args.model, device)

    print("Running mask optimisation (deletion game, paper Eq. 4)...")
    mask, perturbed, init_score, final_score, target_class = optimize_mask(
        model,
        img,
        target_class=None,
        n_iter=args.iter,
        lr=0.1,
        l1_coeff=1e-4,
        tv_coeff=1e-2,
        tv_beta=3,
        blur_sigma=10,
        mask_smooth_sigma=5,
        perturb_type=args.perturb,
        use_jitter=not args.no_jitter,
        jitter_max=4,
        circular_init=not args.no_circular_init,
        verbose=True,
    )

    class_name = get_class_label(target_class)
    print(f"\nTarget class : {class_name}  (index {target_class})")
    print(f"Softmax score: {init_score:.4f}  →  {final_score:.4f}")

    # Normalised suppression (paper footnote 4)
    p0, pb = baseline_scores(model, img, target_class, perturb_type=args.perturb)
    norm_supp = normalized_score(final_score, p0, pb)
    print(f"Normalised suppression p′: {norm_supp:.4f}  (1.0 = complete suppression)")

    out_dir = Path(args.out)
    prefix = img_path.stem
    save_explanations(mask, img, perturbed, out_dir, prefix=prefix)
    print(f"\nSaved to {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
