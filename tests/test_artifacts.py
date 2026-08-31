import unittest
from pathlib import Path

from src.detector import FEATURE_DIMS, load_and_validate_config
from src.frequency import FFT_FEATURE_DIM


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REQUIRED_ARTIFACTS = (
    ARTIFACTS_DIR / "selected_spatial_probe.joblib",
    ARTIFACTS_DIR / "frequency_probe.joblib",
    ARTIFACTS_DIR / "selected_config.json",
)


@unittest.skipUnless(
    all(path.is_file() for path in REQUIRED_ARTIFACTS),
    "trained Colab artifacts have not been copied into artifacts/ yet",
)
class TrainedArtifactContractTests(unittest.TestCase):
    def test_probe_dimensions_match_selected_configuration(self):
        import joblib

        config = load_and_validate_config(ARTIFACTS_DIR / "selected_config.json")
        spatial_model = joblib.load(ARTIFACTS_DIR / "selected_spatial_probe.joblib")
        frequency_model = joblib.load(ARTIFACTS_DIR / "frequency_probe.joblib")

        self.assertEqual(
            int(spatial_model.n_features_in_),
            FEATURE_DIMS[config["feature_variant"]],
        )
        self.assertEqual(int(frequency_model.n_features_in_), FFT_FEATURE_DIM)
        self.assertTrue(config["use_fft_fusion"])


if __name__ == "__main__":
    unittest.main()
