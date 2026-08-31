import json
import tempfile
import unittest
from pathlib import Path

from src.detector import FEATURE_DIMS, load_and_validate_config, prepare_rgb_image


VALID_CONFIG = {
    "model_name": "facebook/dinov2-base",
    "feature_variant": "last4_cls_mean",
    "use_fft_fusion": True,
    "fusion_alpha": 0.7,
    "threshold": 0.53,
}


class DetectorConfigTests(unittest.TestCase):
    def write_config(self, directory: Path, config: dict) -> Path:
        path = directory / "selected_config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_valid_selected_configuration(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = self.write_config(Path(directory_name), VALID_CONFIG)
            config = load_and_validate_config(path)

        self.assertEqual(config["feature_variant"], "last4_cls_mean")
        self.assertEqual(FEATURE_DIMS[config["feature_variant"]], 3_840)

    def test_missing_key_fails(self):
        with tempfile.TemporaryDirectory() as directory_name:
            config = dict(VALID_CONFIG)
            config.pop("fusion_alpha")
            path = self.write_config(Path(directory_name), config)
            with self.assertRaisesRegex(ValueError, "missing keys"):
                load_and_validate_config(path)

    def test_out_of_range_probability_setting_fails(self):
        with tempfile.TemporaryDirectory() as directory_name:
            config = dict(VALID_CONFIG, threshold=1.5)
            path = self.write_config(Path(directory_name), config)
            with self.assertRaisesRegex(ValueError, "threshold"):
                load_and_validate_config(path)

    def test_fusion_flag_must_be_json_boolean(self):
        with tempfile.TemporaryDirectory() as directory_name:
            config = dict(VALID_CONFIG, use_fft_fusion="false")
            path = self.write_config(Path(directory_name), config)
            with self.assertRaisesRegex(ValueError, "JSON boolean"):
                load_and_validate_config(path)

    def test_missing_configuration_has_actionable_message(self):
        with self.assertRaisesRegex(FileNotFoundError, "Copy selected_config.json"):
            load_and_validate_config("does-not-exist.json")

    def test_bad_image_has_path_in_error(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "broken.png"
            path.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "broken.png"):
                prepare_rgb_image(path)


if __name__ == "__main__":
    unittest.main()
