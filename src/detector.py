"""Frozen DINOv2-Base and FFT probability-fusion detector."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .frequency import FFT_FEATURE_DIM, fft_radial_features

FEATURE_DIMS = {
    "cls": 768,
    "cls_mean": 1_536,
    "last4_cls_mean": 3_840,
}
REQUIRED_CONFIG_KEYS = {
    "model_name",
    "feature_variant",
    "use_fft_fusion",
    "fusion_alpha",
    "threshold",
}


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    """Load the Colab-selected configuration and validate its public contract."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing selected model configuration: {config_path}. "
            "Copy selected_config.json from the completed DINOv2 Colab run."
        )
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    missing = REQUIRED_CONFIG_KEYS.difference(config)
    if missing:
        raise ValueError(f"selected_config.json is missing keys: {sorted(missing)}")

    feature_variant = str(config["feature_variant"])
    if feature_variant not in FEATURE_DIMS:
        raise ValueError(
            f"Unsupported feature_variant {feature_variant!r}; "
            f"expected one of {sorted(FEATURE_DIMS)}"
        )
    if not isinstance(config["model_name"], str) or not config["model_name"].strip():
        raise ValueError("model_name must be a non-empty string")

    threshold = float(config["threshold"])
    fusion_alpha = float(config["fusion_alpha"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 <= fusion_alpha <= 1.0:
        raise ValueError("fusion_alpha must be between 0 and 1")

    if not isinstance(config["use_fft_fusion"], bool):
        raise ValueError("use_fft_fusion must be a JSON boolean")

    config["feature_variant"] = feature_variant
    config["threshold"] = threshold
    config["fusion_alpha"] = fusion_alpha
    return config


def prepare_rgb_image(image_or_path: Image.Image | str | Path) -> Image.Image:
    """Return an independent RGB PIL image without leaving a file open."""

    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB").copy()
    image_path = Path(image_or_path)
    try:
        with Image.open(image_path) as image_file:
            return image_file.convert("RGB").copy()
    except Exception as exc:
        raise ValueError(f"Could not decode image {image_path}: {exc}") from exc


class SIDDetector:
    """Load the validated artifacts and return AIGC probabilities for images."""

    def __init__(
        self,
        artifacts_dir: str | Path,
        *,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        try:
            import joblib
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Inference dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        self._torch = torch
        self.batch_size = int(batch_size)
        self.artifacts_dir = Path(artifacts_dir)
        self.config = load_and_validate_config(
            self.artifacts_dir / "selected_config.json"
        )
        self.device = self._resolve_device(device)

        spatial_path = self.artifacts_dir / "selected_spatial_probe.joblib"
        frequency_path = self.artifacts_dir / "frequency_probe.joblib"
        if not spatial_path.is_file():
            raise FileNotFoundError(f"Missing spatial probe: {spatial_path}")
        if self.config["use_fft_fusion"] and not frequency_path.is_file():
            raise FileNotFoundError(f"Missing frequency probe: {frequency_path}")

        self.spatial_model = joblib.load(spatial_path)
        self.frequency_model = (
            joblib.load(frequency_path)
            if self.config["use_fft_fusion"]
            else None
        )
        self._validate_probe_dimensions()

        model_name = self.config["model_name"]
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(self.device)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size != 768:
            raise ValueError(
                f"Expected DINOv2-Base hidden size 768, received {hidden_size}"
            )

    @property
    def threshold(self) -> float:
        return float(self.config["threshold"])

    def _resolve_device(self, requested: str):
        requested = str(requested).strip().lower()
        if requested == "auto":
            requested = "cuda" if self._torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        try:
            return self._torch.device(requested)
        except Exception as exc:
            raise ValueError(f"Invalid device {requested!r}") from exc

    def _validate_probe_dimensions(self) -> None:
        feature_variant = self.config["feature_variant"]
        expected_spatial = FEATURE_DIMS[feature_variant]
        actual_spatial = getattr(self.spatial_model, "n_features_in_", None)
        if actual_spatial is not None and int(actual_spatial) != expected_spatial:
            raise ValueError(
                "Spatial probe dimension mismatch: "
                f"expected {expected_spatial}, got {actual_spatial}"
            )

        if self.frequency_model is not None:
            actual_frequency = getattr(self.frequency_model, "n_features_in_", None)
            if (
                actual_frequency is not None
                and int(actual_frequency) != FFT_FEATURE_DIM
            ):
                raise ValueError(
                    "Frequency probe dimension mismatch: "
                    f"expected {FFT_FEATURE_DIM}, got {actual_frequency}"
                )

    def _autocast(self):
        if self.device.type == "cuda":
            return self._torch.autocast(
                device_type="cuda",
                dtype=self._torch.float16,
            )
        return contextlib.nullcontext()

    def _features_from_output(self, output):
        final_tokens = output.last_hidden_state
        final_cls = final_tokens[:, 0, :]
        patch_mean = final_tokens[:, 1:, :].mean(dim=1)
        last_four_cls = [hidden[:, 0, :] for hidden in output.hidden_states[-4:]]
        variants = {
            "cls": final_cls,
            "cls_mean": self._torch.cat([final_cls, patch_mean], dim=1),
            "last4_cls_mean": self._torch.cat(
                [*last_four_cls, patch_mean],
                dim=1,
            ),
        }
        selected = variants[self.config["feature_variant"]]
        expected_dim = FEATURE_DIMS[self.config["feature_variant"]]
        if selected.ndim != 2 or selected.shape[1] != expected_dim:
            raise ValueError(
                f"Unexpected DINO feature shape {tuple(selected.shape)}; "
                f"expected (*, {expected_dim})"
            )
        return selected

    def _dino_features(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode(), self._autocast():
            output = self.backbone(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        features = self._features_from_output(output).float().cpu().numpy()
        if not np.isfinite(features).all():
            raise ValueError("DINO produced non-finite features")
        return features

    def predict_proba(
        self,
        images: Sequence[Image.Image | str | Path],
    ) -> np.ndarray:
        """Return one AIGC probability in [0, 1] for every input image."""

        if len(images) == 0:
            raise ValueError("At least one image is required")

        probability_chunks: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            prepared = [
                prepare_rgb_image(image)
                for image in images[start : start + self.batch_size]
            ]
            dino_features = self._dino_features(prepared)
            spatial_probability = self.spatial_model.predict_proba(
                dino_features
            )[:, 1]

            if self.frequency_model is None:
                final_probability = spatial_probability
            else:
                fft_features = np.vstack(
                    [fft_radial_features(image) for image in prepared]
                )
                frequency_probability = self.frequency_model.predict_proba(
                    fft_features
                )[:, 1]
                alpha = self.config["fusion_alpha"]
                final_probability = (
                    alpha * spatial_probability
                    + (1.0 - alpha) * frequency_probability
                )
            probability_chunks.append(np.asarray(final_probability, dtype=np.float64))

        probabilities = np.concatenate(probability_chunks)
        if probabilities.shape != (len(images),) or not np.isfinite(probabilities).all():
            raise ValueError("Detector returned invalid probabilities")
        return np.clip(probabilities, 0.0, 1.0)

    def predict(
        self,
        images: Sequence[Image.Image | str | Path],
    ) -> np.ndarray:
        """Return thresholded binary predictions (0=REAL, 1=AIGC)."""

        return (self.predict_proba(images) >= self.threshold).astype(np.int64)
