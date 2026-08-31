# Error analysis

The locked SID holdout contains 400 images: 200 REAL and 200 AIGC. On clean
images, the final DINOv2-Base + FFT detector correctly classified 383 images.
It flagged 5 REAL images as AIGC and accepted 12 AIGC images as REAL.

## Observed trade-offs

- REAL recall remained between 94.0% and 97.5% across the five evaluated
  conditions. The 0.5x resize condition caused the largest drop in REAL recall.
- JPEG-70 produced the lowest overall transformed accuracy at 94.5%, mainly
  because 14 AIGC images were accepted as REAL.
- JPEG-30 performed slightly better than JPEG-70 on this sample. This should not
  be interpreted as a general monotonic relationship between compression and
  performance; it may reflect dataset-specific frequency signatures.
- Frequency features substantially improved SID performance, but they may also
  learn generator- or dataset-specific artifacts. External-generator testing is
  required before making production-generalisation claims.

Representative image-level false-positive and false-negative examples should be
selected from `final_holdout_predictions.csv` after that Drive artifact and the
corresponding manifest are added. Images must only be published if their dataset
licence permits redistribution.
