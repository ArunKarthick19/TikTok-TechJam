"""Command-line interface for directory-to-JSON AIGC confidence scoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from src.detector import SIDDetector

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def discover_images(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    images = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.as_posix().casefold(),
    )
    if not images:
        raise ValueError(f"No supported images found under: {root}")
    return images


def write_predictions_atomic(
    output_path: str | Path,
    image_paths: Sequence[Path],
    probabilities: Sequence[float],
) -> None:
    if len(image_paths) != len(probabilities):
        raise ValueError("Image and probability counts do not match")

    records = [
        {"image_path": path.as_posix(), "pred": float(probability)}
        for path, probability in zip(image_paths, probabilities)
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(records, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score every supported image in a directory for AIGC likelihood."
    )
    parser.add_argument("--input_dir", required=True, help="Directory of input images")
    parser.add_argument("--output", required=True, help="Destination JSON file")
    parser.add_argument(
        "--artifacts_dir",
        default=str(DEFAULT_ARTIFACTS_DIR),
        help="Directory containing the two probes and selected_config.json",
    )
    parser.add_argument(
        "--batch_size",
        type=positive_integer,
        default=16,
        help="DINO inference batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific CUDA device such as cuda:0",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    image_paths = discover_images(args.input_dir)
    detector = SIDDetector(
        args.artifacts_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    probabilities = detector.predict_proba(image_paths)
    write_predictions_atomic(args.output, image_paths, probabilities)
    print(f"Scored {len(image_paths)} image(s). Saved {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
