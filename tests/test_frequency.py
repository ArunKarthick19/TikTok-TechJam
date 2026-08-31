import unittest

import numpy as np
from PIL import Image

from src.frequency import FFT_FEATURE_DIM, fft_radial_features


class FrequencyFeatureTests(unittest.TestCase):
    def test_features_are_finite_deterministic_and_expected_size(self):
        pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        image = Image.fromarray(pixels, mode="RGB")

        first = fft_radial_features(image)
        second = fft_radial_features(image)

        self.assertEqual(first.shape, (FFT_FEATURE_DIM,))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_non_rgb_image_is_supported(self):
        image = Image.new("L", (20, 30), color=127)
        features = fft_radial_features(image)
        self.assertEqual(features.shape, (FFT_FEATURE_DIM,))

    def test_invalid_bin_count_fails(self):
        with self.assertRaises(ValueError):
            fft_radial_features(Image.new("RGB", (8, 8)), bins=0)


if __name__ == "__main__":
    unittest.main()
