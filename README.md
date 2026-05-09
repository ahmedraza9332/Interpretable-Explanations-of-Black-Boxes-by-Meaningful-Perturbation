# Meaningful Perturbation — Interpretable Explanations of Black Boxes

Full implementation of **Fong & Vedaldi (2017)**:  
*Interpretable Explanations of Black Boxes by Meaningful Perturbation*  
Paper: <https://arxiv.org/abs/1704.03296>

---

## What this implements

The codebase implements **Equation 4** of the paper (deletion game with anti-artifact regularisation):

```
min_{m ∈ [0,1]}  λ1·||1-m||_1  +  λ2·Σ||∇m(u)||^β_β  +  E_τ[f_c(Φ(x0(·-τ), m))]
```

where:
- `m` is a low-resolution mask (28×28), upsampled and smoothed to image size
- `Φ` is the perturbation operator (blur / constant / noise)
- `λ1 = 1e-4`, `λ2 = 1e-2`, `β = 3`  (paper §4.3 defaults)
- Jitter `τ` is drawn uniformly from `[0, 4)` per iteration (paper §4.3)
- Mask initialised as **smallest centered circular mask achieving 99% suppression** (paper §4.3)

All three perturbation types from **Section 4.1** are implemented:
- `blur` — `Φ = m·x + (1-m)·G_{σ0}*x`,   σ₀ = 10  ← paper default
- `constant` — `Φ = m·x + (1-m)·μ₀`
- `noise` — `Φ = m·x + (1-m)·η`

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Single-image explanation

```bash
python explain.py images/coffee.jpg
python explain.py images/cat.jpg --model googlenet --iter 300 --out output/
```

Options:

| Flag | Default | Paper value |
|---|---|---|
| `--model` | `googlenet` | GoogLeNet (§4.3) |
| `--iter` | `300` | 300 |
| `--perturb` | `blur` | blur |
| `--no-jitter` | off | jitter enabled |
| `--no-circular-init` | off | circular init enabled |

Output (saved to `output/`):

| File | Description |
|---|---|
| `*_original.png` | Input image |
| `*_mask.png` | Learned mask (white = keep, black = perturb) |
| `*_heatmap.png` | Saliency overlay (red = most important regions) |
| `*_perturbed.png` | Image with salient region blurred |

Printed metrics per image:
- Raw softmax score before/after: `init_score → final_score`
- **Normalised suppression** `p′ = (p0 - p) / (p0 - pb)` (paper footnote 4)

---

## Quantitative evaluation (paper Section 5)

```bash
# Demo on sample images
python evaluate.py --images images/ --out eval_output/

# Full §5.2 / §5.3 (needs ~5 000 ImageNet val images)
python evaluate.py --images /data/imagenet/val/ --n 5000 \
                   --model googlenet --out eval_output/

# §5.6 localization (needs GT bboxes JSON)
python evaluate.py --images /data/imagenet/val/ --n 50000 \
                   --bboxes val_bboxes.json --model googlenet --out eval_output/
```

### §5.2 — Deletion region representativeness (Figure 7)

Binary masks derived from the learned mask (at 20 thresholds α ∈ [0, 0.95]) are
applied with all three perturbation types.  The normalised suppression score
`p′ = (p0 - p) / (p0 - pb)` is reported vs α; it should rise quickly as more of
the salient region is deleted.

### §5.3 — Minimality of deletions (Figure 8)

The saliency heatmap is thresholded at h ∈ [0, 1], a tight bounding box is fitted
around the resulting region, and the image inside the box is blurred.  Reports the
smallest box area (as % of image) that achieves 80 / 90 / 95 / 99% suppression.

### §5.6 — Localization error (Table 1)

Requires `--bboxes` JSON (`{"filename.jpg": [y1, x1, y2, x2], ...}` in 224×224 space).
Three thresholding strategies are evaluated:

| Method | Paper (Mask‡) | Strategy |
|---|---|---|
| Value-α | 44.0% | threshold by intensity |
| Energy-α | 43.1% | threshold by % of total energy |
| Mean-α | 43.2% | threshold by α × mean intensity |

IoU ≥ 0.5 is counted as correct localisation.

### Output files

| File | Contents |
|---|---|
| `eval_output/results.json` | Per-image scores and curves |
| `eval_output/summary.json` | Aggregated §5.2, §5.3, §5.6 numbers |
| `eval_output/masks/*.png` | Per-image masks and heatmaps |

---

## Project structure

```
Project/
├── explain.py            # Single-image entry point
├── evaluate.py           # Quantitative evaluation (§5.2, §5.3, §5.6)
├── mask_optimizer.py     # Deletion game (Eq. 4) + circular init
├── perturbation.py       # Blur / constant / noise operators (§4.1)
├── metrics.py            # Paper evaluation metrics (§5.2, §5.3, §5.6)
├── utils.py              # Preprocessing, model loading, visualisation
├── fetch_sample_images.py
├── requirements.txt
├── images/               # Sample images
└── output/               # Generated explanations
```

---

## Paper alignment checklist

| Paper component | Status |
|---|---|
| Equation 4 (deletion game loss) | ✅ implemented |
| Circular-mask initialisation (§4.3) | ✅ implemented |
| Blur / constant / noise perturbations (§4.1) | ✅ implemented |
| Low-res mask (28×28) + bilinear upsample + Gaussian smooth | ✅ implemented |
| TV regularisation β=3 | ✅ implemented |
| Jitter τ∈[0,4) one-sided uniform (§4.3) | ✅ implemented |
| Adam, lr=0.1, λ1=1e-4, λ2=1e-2 (§4.3) | ✅ implemented |
| Normalised suppression score (footnote 4) | ✅ implemented |
| §5.2 deletion representativeness curves | ✅ implemented |
| §5.3 minimality / smallest bounding box | ✅ implemented |
| §5.6 localization error (Table 1) | ✅ implemented |
| §5.5 FGSM adversarial generation | ✅ implemented (metrics.py) |
