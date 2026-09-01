# Robustness Evaluation Summary

Final model: **CLIP (frozen) + FFT radial features, gated fusion** (α = 0.55).
Evaluated on the SID_Set validation split (200 real / 400 fake, drawn from
`saberzl/SID_Set`) against all 12 conditions specified in the problem
statement's transform table.

## Full 12-condition results

| Condition | Accuracy | AUC |
|---|---|---|
| Clean | 0.9383 | 0.9861 |
| JPEG q90 | 0.9383 | 0.9802 |
| JPEG q50 | 0.9433 | 0.9858 |
| JPEG q30 | 0.9417 | 0.9882 |
| Gaussian blur σ=0.5 (blur_sigma1) | 0.8383 | 0.9727 |
| Gaussian blur σ=1.0 (blur_sigma2) | 0.9217 | 0.9790 |
| Resize 0.5× | 0.9133 | 0.9854 |
| Resize 0.25× | 0.9167 | 0.9808 |
| Gaussian noise σ=0.05 | 0.8500 | 0.9447 |
| Gaussian noise σ=0.10 | 0.8133 | 0.9106 |
| Color jitter (±20%) | 0.9367 | 0.9847 |
| Center crop 80% | 0.9683 | 0.9928 |

**Final combined score = 0.5 × Clean AUC + 0.5 × mean(robust AUC) = 0.9796**

Every one of the 12 conditions scores above 0.91 AUC — no condition
collapses toward random-guess performance (0.5 AUC).

## How we got here

An earlier version of this architecture (CLIP + FFT with simple feature
*concatenation*, trained only on clean images) scored far worse on this
same condition set — center crop 80% specifically dropped to **0.8437
AUC**, its weakest result, because FFT's sensitivity to global image
structure is disrupted by cropping. Retraining with degradation-consistent
augmentation (randomly applying the transforms above during training, not
just testing) fixed this: crop recovered to **0.9928 AUC**, its best
result. See `error_analysis.md` for the full before/after comparison and
the reasoning behind it.

## Comparison against alternatives tested

| Architecture | Clean AUC | Combined Score |
|---|---|---|
| CLIP + FFT (concatenation) | 0.9700 | 0.9506 |
| DINOv2 + FFT (gated fusion) | 0.9819 | 0.9774 |
| **CLIP + FFT (gated fusion) — final** | **0.9861** | **0.9796** |

CLIP + gated fusion outperformed the DINOv2 variant while also showing
meaningfully more stable accuracy across conditions (DINOv2's accuracy
dropped as low as 0.46–0.68 on blur/resize despite a similarly high AUC,
indicating a threshold-calibration issue that CLIP's fusion did not share).

PRNU-style sensor-noise features, multi-scale FFT, and azimuthal FFT
averaging were also tested and did not beat this configuration — see
`error_analysis.md` for details on each.
