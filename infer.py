"""
Required deliverable: takes an image directory, outputs JSON with
image_path and pred (confidence score, 0=real, 1=AI-generated) for each image.

Usage:
    python infer.py --input_dir /path/to/images --output predictions.json --fusion_alpha 0.XX
"""

import argparse
import json
import os

import joblib
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
FFT_BINS = 32


def get_embedding(output):
    if isinstance(output, torch.Tensor):
        return output
    elif hasattr(output, "image_embeds"):
        return output.image_embeds
    elif hasattr(output, "pooler_output"):
        return output.pooler_output
    else:
        raise ValueError(f"Unexpected output type: {type(output)}")


def fft_radial_features(image, bins=FFT_BINS):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width, _ = array.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - (height - 1) / 2.0) ** 2 + (xx - (width - 1) / 2.0) ** 2)
    radius = radius / max(float(radius.max()), 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    features = []
    for channel in range(3):
        magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(array[:, :, channel]))))
        for i in range(bins):
            if i == bins - 1:
                mask = (radius >= edges[i]) & (radius <= edges[i + 1])
            else:
                mask = (radius >= edges[i]) & (radius < edges[i + 1])
            features.append(float(magnitude[mask].mean()) if mask.any() else 0.0)
    return np.asarray(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--spatial_model", default="spatial_model.joblib")
    parser.add_argument("--frequency_model", default="frequency_model.joblib")
    parser.add_argument("--fusion_alpha", type=float, default=0.55)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    print("Loading trained models...")
    spatial_model = joblib.load(args.spatial_model)
    frequency_model = joblib.load(args.frequency_model)

    image_paths = [
        os.path.join(args.input_dir, f)
        for f in sorted(os.listdir(args.input_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"Found {len(image_paths)} images in {args.input_dir}")

    results = []

    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        images, valid_paths = [], []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"Skipping unreadable image {p}: {e}")
                results.append({"image_path": p, "pred": None})

        if not images:
            continue

        inputs = clip_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            output = clip_model.get_image_features(**inputs)
        clip_features = get_embedding(output).cpu().numpy()

        fft_features = np.stack([fft_radial_features(img) for img in images])

        spatial_probs = spatial_model.predict_proba(clip_features)[:, 1]
        freq_probs = frequency_model.predict_proba(fft_features)[:, 1]
        fused_probs = args.fusion_alpha * spatial_probs + (1 - args.fusion_alpha) * freq_probs

        for p, prob in zip(valid_paths, fused_probs):
            results.append({"image_path": p, "pred": float(prob)})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
