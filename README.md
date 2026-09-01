# Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026 — Track 5 submission.

## Project Overview

This project detects whether an image is real or AI-generated, with a specific
focus on staying accurate after the kinds of transformations images undergo in
real-world redistribution: JPEG compression, blurring, resizing, noise, color
adjustment, and cropping.

Most naive detectors perform well on clean images but collapse once an image
has been compressed, cropped, or reposted — exactly the conditions synthetic
images actually face on real platforms. Our goal was to build a detector that
holds up under those conditions, not just in a lab setting.

### Architecture

Our final model combines two complementary signals through **gated probability
fusion**:

1. **Spatial branch (CLIP):** a frozen, pretrained CLIP ViT-B/32 model
   (151M parameters) extracts a 512-dimensional semantic embedding from each
   image. CLIP was never fine-tuned — we only train a lightweight classifier
   (StandardScaler + Logistic Regression) on top of its frozen embeddings.
2. **Frequency branch (FFT):** we compute a 2D FFT of the image and extract
   radially-averaged magnitude features (32 bins × 3 color channels = 96
   features), capturing spectral artifacts that generative models leave
   behind but that CLIP's semantic features don't see.
3. **Gated fusion:** rather than concatenating both feature sets into one
   classifier, we train two *separate* classifiers (one per branch) and
   combine their output probabilities with a tuned weight:
   `final_score = alpha * P(fake | CLIP) + (1 - alpha) * P(fake | FFT)`.
   Our tuned `alpha = 0.55`.

Hyperparameters (regularization strength, class weighting, and the fusion
weight `alpha`) were selected via grouped cross-validation on the training
split — not hand-picked.

### Why this architecture

We arrived here through a genuinely iterative process, not by picking one
approach upfront:

- We first built CLIP + FFT with simple feature **concatenation**. Testing
  against the organizer's full transform suite revealed the model had
  gotten *worse* at detecting cropped images specifically (AUC dropped to
  0.84) even though overall performance improved. We diagnosed this as FFT's
  sensitivity to global image structure, which cropping disrupts. Retraining
  with degradation-consistent augmentation (randomly applying the organizer's
  transforms during training) fixed this — crop performance recovered from
  our worst condition to our best (0.95 AUC).
- We then tested **gated fusion** instead of concatenation, and separately
  tested **DINOv2** as an alternative backbone. Gated fusion improved our
  combined score to 0.98 regardless of backbone; isolating the two variables
  showed the fusion *technique* was the larger driver of improvement, not the
  backbone choice — CLIP + gated fusion (0.98) slightly outperformed
  DINOv2 + gated fusion (0.977), with meaningfully more stable accuracy
  across conditions.
- We also tested PRNU-style sensor-noise features, multi-scale FFT, azimuthal
  FFT averaging, and contrastive pretraining as further additions. None beat
  our final CLIP + FFT (gated fusion) configuration; each is discussed in
  `error_analysis.md`.

## Results

**Final combined score (0.5 × clean AUC + 0.5 × average robust AUC): 0.9796**

See `robustness_table.md` for the full 12-condition breakdown and
`error_analysis.md` for false-positive/negative examples and a discussion of
out-of-distribution generalization.

## Development Tools

- Google Colab (T4 GPU) for training and experimentation
- Local Jupyter/VS Code for inference script development and testing
- Kaggle (for early prototyping on the CIFAKE dataset)

## Models

- **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) — frozen pretrained
  backbone, via Hugging Face Transformers
- No paid APIs were used; all models run locally/on-GPU

## Libraries and Frameworks

- `transformers` (Hugging Face) — CLIP model and processor
- `torch` — model inference
- `scikit-learn` — classifier training, cross-validation, scaling
- `numpy`, `scipy` — FFT feature extraction
- `datasets` (Hugging Face) — SID_Set streaming
- `Pillow` — image loading and transforms
- `joblib` — model serialization

## Datasets

- **SID_Set** (`saberzl/SID_Set` on Hugging Face) — primary training and
  evaluation dataset (real / fully-synthetic / tampered images)
- **CIFAKE** (Kaggle) — used during early prototyping to validate the
  pipeline end-to-end before scaling to SID_Set

## Setup and Installation

```bash
git clone https://github.com/ArunKarthick19/TikTok-TechJam.git
cd TikTok-TechJam
pip install -r requirements.txt
```

Requires Python 3.10+. A CUDA-capable GPU is recommended but not required
(the script will fall back to CPU automatically).

## Usage

```bash
python infer.py --input_dir /path/to/images --output predictions.json --fusion_alpha 0.55
```

**Arguments:**
- `--input_dir` (required): directory of `.jpg`/`.jpeg`/`.png` images to classify
- `--output` (default: `predictions.json`): where to write results
- `--fusion_alpha` (default: `0.55`): weight toward the CLIP branch in the
  gated fusion (our tuned value)
- `--batch_size` (default: `32`): inference batch size

**Output format:**
```json
[
  {"image_path": "images/photo1.jpg", "pred": 0.83},
  {"image_path": "images/photo2.jpg", "pred": 0.12}
]
```
`pred` is a confidence score between 0 and 1 that the image is AI-generated.

**Note:** the first run downloads CLIP's weights from Hugging Face
(~350MB) and takes roughly 2 minutes; subsequent runs use the cached
weights and are faster.

### Viewing results with human-readable labels

`infer.py` outputs raw confidence scores as required by the problem
statement. For convenience, `view_results.py` reads the same
`predictions.json` and prints a human-readable label alongside each score:

```bash
python view_results.py
```

```python
import json

with open("predictions.json") as f:
    results = json.load(f)

for r in results:
    label = "AI-generated" if r["pred"] >= 0.5 else "Real"
    print(f"{r['image_path']}: {r['pred']:.3f} ({label})")
```

This is a convenience script for demos/manual review only — it is not part
of the required deliverable and is not needed to reproduce our results.

## Steps to Reproduce Our Results

1. Load SID_Set via `datasets.load_dataset("saberzl/SID_Set", split="train", streaming=True)`
2. Extract CLIP embeddings and FFT radial features for a balanced sample
   (we used 500 images/class for training, 200/class for validation)
3. Apply the organizer's transform suite as training-time augmentation
   (random JPEG compression, blur, resize, noise, color jitter, or crop per
   image)
4. Train two `StandardScaler + LogisticRegression` pipelines — one on CLIP
   features, one on FFT features — with hyperparameters selected via 3-fold
   grouped cross-validation
5. Tune the fusion weight `alpha` by sweeping `[0, 1]` in steps of 0.05 and
   selecting the value that maximizes validation AUC
6. Evaluate on the full 12-condition transform suite (see
   `robustness_table.md`)

The full experiment notebook (CLIP + FFT gated fusion — feature extraction,
cross-validation, fusion tuning, and the full robustness sweep) is available
here: **[add your Colab share link]**. We ran this in Colab rather than
committing raw `.ipynb` files, since it depends on a GPU runtime; the link
is set to allow anyone with it to open and run the notebook.

## Limitations and Future Work

- **Out-of-distribution generalization:** our model performs strongly on
  SID_Set (0.98 combined score) but shows reduced accuracy on images from
  unrelated sources (e.g. stock photography, arbitrary internet images) —
  see `error_analysis.md` for a detailed breakdown. We believe this stems
  from SID_Set's real images being less "polished" than professional stock
  photography, causing the model to associate visual polish with
  AI-generation.
- **Threshold calibration is domain-dependent:** our default 0.5 threshold
  is well-calibrated for SID_Set, but a higher threshold (~0.7) performed
  better on our small out-of-distribution test set. A production deployment
  would benefit from domain-specific threshold calibration rather than a
  single fixed cutoff — our script already outputs continuous scores to
  support this.
- **Sample size:** our training/evaluation used a subsample of SID_Set
  (1,500 train / 600 validation images) rather than the full 300K-image
  dataset, due to hackathon time constraints. Results may improve with more
  training data.
- **Untested avenues:** we did not have time to properly evaluate a
  diffusion-reconstruction-error (DIRE-style) detection branch, which
  research suggests could add a further complementary signal; this remains
  a promising direction for future work.
- **Given more time**, we would: build proper
  domain-adaptive threshold calibration.

## Team Contributions

- **Arun Karthick:** CLIP + FFT baseline architecture, degradation-consistent
  training, robustness evaluation pipeline, inference script, error analysis,
  out-of-distribution testing, documentation
- **Wang Jiawei:** DINOv2 comparison architecture, gated fusion technique,
  cross-validation hyperparameter selection
- **Derek Qua:** Early backbone exploration (SigLIP), initial project
  setup, ongoing support across the pipeline
