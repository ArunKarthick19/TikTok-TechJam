import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from predict import discover_images, main, positive_integer, write_predictions_atomic


class PredictCliTests(unittest.TestCase):
    def test_recursive_discovery_is_filtered_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            nested = root / "nested"
            nested.mkdir()
            Image.new("RGB", (4, 4)).save(root / "B.PNG")
            Image.new("RGB", (4, 4)).save(nested / "a.jpg")
            (root / "notes.txt").write_text("ignore me", encoding="utf-8")

            relative_names = [
                path.relative_to(root).as_posix() for path in discover_images(root)
            ]

        self.assertEqual(relative_names, ["B.PNG", "nested/a.jpg"])

    def test_empty_directory_fails(self):
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ValueError, "No supported images"):
                discover_images(directory_name)

    def test_atomic_json_has_exact_schema(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            output = root / "output" / "predictions.json"
            images = [Path("images/a.jpg"), Path("images/b.png")]
            write_predictions_atomic(output, images, [0.125, 0.875])
            records = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(records), 2)
        self.assertEqual(set(records[0]), {"image_path", "pred"})
        self.assertEqual(records[0]["image_path"], "images/a.jpg")
        self.assertEqual(records[1]["pred"], 0.875)

    def test_positive_batch_size_validation(self):
        self.assertEqual(positive_integer("4"), 4)
        with self.assertRaises(Exception):
            positive_integer("0")

    def test_main_writes_predictions_with_injected_detector(self):
        class FakeDetector:
            def __init__(self, *args, **kwargs):
                pass

            def predict_proba(self, images):
                return np.asarray([0.75] * len(images))

        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (8, 8)).save(image_dir / "sample.png")
            output = root / "predictions.json"

            with patch("predict.SIDDetector", FakeDetector):
                exit_code = main(
                    [
                        "--input_dir",
                        str(image_dir),
                        "--output",
                        str(output),
                    ]
                )
            records = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(records[0]["pred"], 0.75)

    def test_failure_does_not_create_partial_output(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (8, 8)).save(image_dir / "sample.png")
            output = root / "predictions.json"

            with patch("predict.SIDDetector", side_effect=RuntimeError("failed")):
                exit_code = main(
                    [
                        "--input_dir",
                        str(image_dir),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
