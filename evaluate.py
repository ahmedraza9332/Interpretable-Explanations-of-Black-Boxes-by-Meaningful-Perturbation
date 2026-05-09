"""
Evaluation script: Fong & Vedaldi (2017) paper experiments.

Runs the Meaningful Perturbation mask optimisation on a set of images and
reports the three quantitative experiments from Section 5 of the paper:

    §5.2  Deletion region representativeness  (suppression curves, Fig. 7)
    §5.3  Minimality of deletions             (smallest bbox, Fig. 8)
    §5.6  Localization error                  (Table 1)  ← requires GT bboxes

Usage
-----
# Demo: 5 sample images, no ground-truth labels needed
    python evaluate.py --images images/ --out eval_output/ --model googlenet

# Full §5.2 / §5.3 on an ImageNet-style folder (5 000 images)
    python evaluate.py --images /data/imagenet/val/ --n 5000 \
                       --model googlenet --out eval_output/

# Full §5.6 localization (requires GT bbox JSON, see --bboxes help)
    python evaluate.py --images /data/imagenet/val/ --n 50000 \
                       --bboxes val_bboxes.json --model googlenet --out eval_output/

Bounding-box JSON format (for --bboxes)
---------------------------------------
    {"ILSVRC2012_val_00000001.JPEG": [y1, x1, y2, x2], ...}
    Coordinates are in 224×224 pixel space (after preprocessing resize).

Paper numbers to reproduce (GoogLeNet, blur perturbation)
----------------------------------------------------------
    §5.2: Normalised suppression rises quickly as α increases for all three
          perturbation types (Fig. 7 bottom – qualitative shape).
    §5.3: On 5 000 ImageNet images, smallest bbox achieving 80/90/95/99%
          suppression (Fig. 8 – paper uses GoogLeNet).
    §5.6: Localization error (Table 1):
          Value-α  → 44.0%
          Energy-α → 43.1%
          Mean-α   → 43.2%
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from utils import load_image, load_model, get_class_label, preprocess_tensor
from mask_optimizer import optimize_mask
from metrics import (
    baseline_scores,
    normalized_score,
    suppression_curve,
    minimality_curve,
    minimality_summary,
    localization_error_single,
    aggregate_localization_error,
)


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def collect_images(directory, n=None):
    """Return sorted list of image paths from a flat or class-folder directory."""
    d = Path(directory)
    exts = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}
    paths = [p for p in d.rglob("*") if p.suffix in exts]
    paths.sort()
    if n is not None:
        paths = paths[:n]
    return paths


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(model, img_path, device, args):
    """
    Run optimisation + metrics for one image.

    Returns (result_dict, mask_np, heatmap_np, img_tensor).
    """
    img_np = load_image(str(img_path))
    img = preprocess_tensor(img_np, device)

    mask, _perturbed, init_score, final_score, target_class = optimize_mask(
        model, img,
        target_class=None,
        n_iter=args.iter,
        lr=0.1,
        l1_coeff=1e-4,
        tv_coeff=1e-2,
        tv_beta=3,
        blur_sigma=10,
        mask_smooth_sigma=5,
        perturb_type=args.perturb,
        use_jitter=True,
        jitter_max=4,
        circular_init=True,
        verbose=False,
    )

    mask_np = mask.squeeze().detach().cpu().numpy()      # (H, W), m=1 keep
    heatmap_np = 1.0 - mask_np                           # saliency: high = deleted

    p0, pb = baseline_scores(model, img, target_class, perturb_type=args.perturb)

    result = {
        "image": img_path.name,
        "target_class": int(target_class),
        "class_name": get_class_label(target_class),
        "p0": float(p0),
        "pb": float(pb),
        "init_score": float(init_score),
        "final_score": float(final_score),
        "normalized_final_suppression": float(normalized_score(final_score, p0, pb)),
    }

    # §5.2 Deletion region representativeness
    alphas = np.arange(0.0, 1.0, 0.05)
    sc_alphas, sc_results = suppression_curve(
        model, img, mask_np, target_class, p0, pb,
        alphas=alphas,
        perturb_types=["blur", "constant", "noise"],
        extra_smooth_sigma=10,
    )
    result["sec52_alphas"] = sc_alphas.tolist()
    result["sec52_curves"] = sc_results   # {perturb_type: [scores]}

    # §5.3 Minimality of deletions
    min_thresholds = np.arange(0.0, 1.1, 0.1)
    thresholds, norm_scores, bbox_areas = minimality_curve(
        model, img, heatmap_np, target_class, p0, pb,
        thresholds=min_thresholds,
    )
    result["sec53_thresholds"] = thresholds.tolist()
    result["sec53_norm_scores"] = norm_scores
    result["sec53_bbox_areas"] = bbox_areas
    result["sec53_summary"] = {
        str(k): v
        for k, v in minimality_summary(thresholds, norm_scores, bbox_areas).items()
    }

    return result, mask_np, heatmap_np, img


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def mean_suppression_curves(results):
    """Average §5.2 suppression curves across all images (Fig. 7 bottom)."""
    alphas = results[0]["sec52_alphas"]
    averaged = {}
    for pt in ["blur", "constant", "noise"]:
        curves = [r["sec52_curves"].get(pt, []) for r in results if r.get("sec52_curves")]
        curves = [c for c in curves if len(c) == len(alphas)]
        if curves:
            averaged[pt] = np.mean(curves, axis=0).tolist()
    return alphas, averaged


def mean_minimality(results):
    """Average §5.3 minimality summary across all images (Fig. 8)."""
    agg = {}
    for level_str in ["0.8", "0.9", "0.95", "0.99"]:
        vals = [
            r["sec53_summary"][level_str]
            for r in results
            if r.get("sec53_summary", {}).get(level_str) is not None
        ]
        agg[level_str] = float(np.mean(vals)) if vals else None
    return agg


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(results, loc_errors=None):
    n = len(results)
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY  —  Fong & Vedaldi (2017)")
    print("=" * 65)
    print(f"  Images evaluated : {n}")

    mean_p0 = np.mean([r["p0"] for r in results])
    mean_pb = np.mean([r["pb"] for r in results])
    mean_ns = np.mean([r["normalized_final_suppression"] for r in results])
    print(f"\n  Avg original score p0         : {mean_p0:.4f}")
    print(f"  Avg fully-perturbed score pb  : {mean_pb:.4f}")
    print(f"  Avg normalised suppression    : {mean_ns:.4f}  (1.0 = perfect)")

    # §5.2
    print("\n  §5.2  Deletion region representativeness")
    print("  (Avg normalised suppression at α = 0.5 for binary masks)")
    alphas, avg_curves = mean_suppression_curves(results)
    mid = len(alphas) // 2
    for pt, curve in avg_curves.items():
        print(f"    {pt:10s}: {curve[mid]:.4f}")

    # §5.3
    print("\n  §5.3  Minimality — smallest bbox achieving suppression level")
    for level_str, area in mean_minimality(results).items():
        area_str = f"{area * 100:.1f}% of image area" if area is not None else "not achieved"
        print(f"    {float(level_str)*100:.0f}% suppression : {area_str}")

    # §5.6
    if loc_errors:
        print("\n  §5.6  Localization error  (paper Table 1 target: ~43–44%)")
        for method in ["value", "energy", "mean"]:
            errs = loc_errors.get(method, [])
            if errs:
                _, err_pct = aggregate_localization_error(errs)
                print(f"    {method:8s}-α : {err_pct:.1f}%  (N={len(errs)})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Meaningful Perturbation — Fong & Vedaldi (2017)"
    )
    parser.add_argument("--images", type=str, default="images/",
                        help="Directory of images (flat or ImageNet class-folder).")
    parser.add_argument("--out", type=str, default="eval_output/",
                        help="Output directory for results and masks.")
    parser.add_argument("--model", type=str, default="googlenet",
                        choices=["vgg19", "googlenet", "alexnet"],
                        help="Classifier. Paper uses googlenet.")
    parser.add_argument("--iter", type=int, default=300,
                        help="Optimisation iterations per image (paper: 300).")
    parser.add_argument("--perturb", type=str, default="blur",
                        choices=["blur", "constant", "noise"],
                        help="Perturbation type (paper default: blur).")
    parser.add_argument("--n", type=int, default=None,
                        help="Max number of images to process.")
    parser.add_argument(
        "--bboxes", type=str, default=None,
        help=(
            "JSON mapping image filename → [y1,x1,y2,x2] in 224×224 space. "
            "Required for §5.6 localization error."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {args.model}  |  Perturbation: {args.perturb}  |  Iters: {args.iter}")

    model = load_model(args.model, device)

    gt_bboxes = {}
    if args.bboxes and Path(args.bboxes).exists():
        with open(args.bboxes) as f:
            gt_bboxes = json.load(f)
        print(f"Loaded {len(gt_bboxes)} ground-truth bboxes from {args.bboxes}")

    images = collect_images(args.images, n=args.n)
    if not images:
        print(f"No images found in '{args.images}'.")
        return 1
    print(f"\nFound {len(images)} images.  Running evaluation...\n")

    out_dir = Path(args.out)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    # Per-image error lists for §5.6 aggregation: {method: [[errors_per_alpha], ...]}
    loc_per_image = {"value": [], "energy": [], "mean": []}

    for idx, img_path in enumerate(images):
        t0 = time.time()
        print(f"[{idx+1:>4}/{len(images)}]  {img_path.name:<40}", end="", flush=True)

        try:
            result, mask_np, heatmap_np, img_tensor = process_image(
                model, img_path, device, args
            )
            all_results.append(result)

            # Save mask and heatmap images
            stem = img_path.stem
            cv2.imwrite(
                str(masks_dir / f"{stem}_mask.png"),
                (mask_np * 255).astype(np.uint8),
            )
            heatmap_vis = cv2.applyColorMap(
                (heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            cv2.imwrite(str(masks_dir / f"{stem}_heatmap.png"), heatmap_vis)

            # §5.6 localization (only when GT is available)
            if img_path.name in gt_bboxes:
                gt_bbox = tuple(gt_bboxes[img_path.name])
                alphas_val = np.arange(0.0, 1.0, 0.05)
                alphas_mean = np.arange(0.0, 10.5, 0.5)
                for method, alphas in [
                    ("value", alphas_val),
                    ("energy", alphas_val),
                    ("mean", alphas_mean),
                ]:
                    _best_a, errors = localization_error_single(
                        heatmap_np, gt_bbox,
                        method=method, alphas=alphas, iou_threshold=0.5,
                    )
                    loc_per_image[method].append(errors)

            ns = result["normalized_final_suppression"]
            elapsed = time.time() - t0
            print(f"  norm_supp={ns:.3f}  class={result['class_name'][:20]:<20}  [{elapsed:.1f}s]")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    # Save full JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {out_dir / 'results.json'}")

    if not all_results:
        print("No results to aggregate.")
        return 1

    print_summary(
        all_results,
        loc_errors={k: v for k, v in loc_per_image.items() if v},
    )

    # Save aggregate summary JSON
    alphas, avg_curves = mean_suppression_curves(all_results)
    summary = {
        "n_images": len(all_results),
        "model": args.model,
        "perturb_type": args.perturb,
        "sec52_mean_suppression_curves": {"alphas": alphas, "curves": avg_curves},
        "sec53_minimality": mean_minimality(all_results),
    }
    if any(loc_per_image.values()):
        summary["sec56_localization"] = {}
        for method, errs in loc_per_image.items():
            if errs:
                best_idx, err_pct = aggregate_localization_error(errs)
                summary["sec56_localization"][method] = {
                    "error_pct": round(err_pct, 2),
                    "n_images": len(errs),
                }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregate summary saved to {out_dir / 'summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
