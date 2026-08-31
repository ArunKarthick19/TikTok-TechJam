# Robust AI-Generated Image Detection

This project detects AI-generated images after common real-world transformations
such as JPEG compression, resizing, and blur. The official detector combines a
frozen DINOv2-Base representation with lightweight frequency-domain features.
CLIP is retained as an experimental baseline.

## Architecture

Each image is processed through two complementary branches:

1. **Spatial-semantic branch:** frozen `facebook/dinov2-base`, represented by
   the CLS tokens from its final four transformer layers plus the final-layer
   mean patch token (3,840 features).
2. **Frequency branch:** native-resolution RGB FFT log-magnitude radial profiles
   with 32 bins per channel (96 features).

Separate standardised logistic-regression probes produce AIGC probabilities.
Their probabilities are combined using the validated fusion weight stored in
`artifacts/selected_config.json`. Inference requires one DINOv2 forward pass and
no backbone fine-tuning or test-time augmentation.

## Results

The final configuration was selected using three-fold grouped cross-validation
on 1,200 SID development images. Hyperparameters, fusion, and threshold were
frozen before one evaluation on a balanced 400-image holdout.

| Condition | Accuracy | Macro F1 | REAL recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Clean | 95.75% | 95.75% | 97.50% | 0.9946 |
| JPEG quality 70 | 94.50% | 94.50% | 96.00% | 0.9909 |
| JPEG quality 30 | 96.00% | 96.00% | 96.00% | 0.9951 |
| Resize 0.5x | 94.75% | 94.75% | 94.00% | 0.9933 |
| Blur sigma 1.0 | 94.75% | 94.75% | 96.00% | 0.9943 |

Mean holdout ROC-AUC was **0.9936** and worst-condition balanced accuracy was
**94.50%**. Detailed values are in `results/final_holdout_metrics.csv`.

## Installation

Python 3.11 or newer is recommended. A CUDA-capable GPU is faster, but CPU
inference is supported.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The DINOv2 backbone is downloaded from Hugging Face on first use.

## Trained artifacts

Place these files from the completed DINOv2 Colab run in `artifacts/`:

- `selected_spatial_probe.joblib`
- `frequency_probe.joblib`
- `selected_config.json`

The program validates the configuration and feature dimensions. It deliberately
fails if artifacts are absent instead of using unvalidated defaults.

## Directory-to-JSON inference

```bash
python predict.py --input_dir ./images --output predictions.json
```

Optional controls:

```bash
python predict.py \
  --input_dir ./images \
  --output predictions.json \
  --batch_size 16 \
  --device auto
```

The command recursively discovers JPG, JPEG, PNG, WEBP, and BMP files. Its JSON
output contains the AIGC probability for every image:

```json
[
  {
    "image_path": "images/example.jpg",
    "pred": 0.9724
  }
]
```

`pred` is a confidence score between 0 and 1, where larger values indicate a
higher likelihood that the image is AI-generated. Unsupported files are
ignored. A corrupt supported image causes the command to fail without writing a
partial result.

## Testing

Run the lightweight deterministic tests without downloading DINOv2:

```bash
python -m unittest discover -s tests -v
```

After installing dependencies and copying the trained artifacts, compare several
standalone probabilities with the Colab predictor. Labels must match and
probabilities should differ by less than `1e-3`.

## Development tools and data

- Development: Google Colab, Jupyter, VS Code
- Models: DINOv2-Base and CLIP ViT-B/32
- Libraries: PyTorch, Hugging Face Transformers, scikit-learn, NumPy, Pillow
- Primary dataset: [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
- Additional baseline dataset: [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

The experiment notebooks and their Colab links are documented under
`notebooks/`.

## Limitations and future work

- The final holdout comes from SID and therefore does not establish performance
  on unseen datasets, generators, or camera pipelines.
- FFT features may exploit dataset-specific generation or encoding artifacts.
- Only five clean/degradation conditions were evaluated for the final model.
- CPU inference is supported but slower than GPU inference.
- Future work should add external-generator evaluation, the remaining official
  transformations, probability calibration, and image-level explainability.

## Team contributions
