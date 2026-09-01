# Error Analysis

## Representative false positives and false negatives (SID_Set)

At our default decision threshold (0.5), the model occasionally misclassifies
real images with unusually high visual polish. One consistent example from
our validation slice:

- **A real image scoring 0.715** (above the 0.5 threshold, so misclassified
  as AI-generated) was the one recurring error across repeated evaluation of
  the same slice. Its score sits closer to the decision boundary than our
  other real-image scores (which clustered around 0.02–0.52), suggesting the
  model found it visually closer to its learned "fake" distribution than
  most real photos.

We did not find a comparably clear pattern among false negatives on SID_Set
itself — the model's AUC (0.986 clean) indicates very few fakes are ranked
below real images in the validation set.

## Out-of-distribution testing (internet-sourced images)

Beyond SID_Set, we tested the final model against a small, independently
assembled set of 20 images from outside our training/validation
distribution: 10 AI-generated images (from ChatGPT, DALL·E, Flux, and
Invoke) and 10 real photographs (from Flickr, Unsplash, and other sources).

**At the default threshold (0.5): 60% accuracy (12/20).**
All 10 AI-generated images were correctly identified. 8 of 10 real photos
were misclassified as AI-generated.

**At threshold 0.7: 90% accuracy (18/20).**
All 10 AI-generated images remained correctly identified. 8 of 10 real
photos were now correctly identified — a clear improvement from raising the
cutoff.

### Interpretation

We confirmed this improvement is genuinely a *threshold-calibration* issue,
not evidence that the model's underlying signal is unreliable: the model's
raw scores still separate real from fake reasonably well on this external
set, but the boundary that works for SID_Set does not transfer directly.

We verified this is not a free win, however: re-running the same threshold
(0.7) against our SID_Set validation slice reduced accuracy from 80% to 70%
on the same 10-image sample (two previously-correct fakes flipped to
misses). **We kept 0.5 as our submitted default**, since it is the
threshold properly validated against our actual training/evaluation
distribution (SID_Set); 0.7 helps specifically for this external,
out-of-distribution case. Because `infer.py` outputs a continuous
confidence score rather than a hard label, a downstream user can apply
whichever threshold suits their own deployment domain.

### Remaining false positives at threshold 0.7

Two real photos were still misclassified even at the higher threshold:

- **A drone/aerial shot** of a highway interchange with strong geometric
  symmetry (score: 0.800)
- **A macro product photograph** of a mineral specimen in a museum display,
  with dramatic, deliberate lighting (score: 0.827)

Both are examples of highly polished, professional photography — sharp
focus, deliberate composition, dramatic lighting. We hypothesize the model
associates this level of visual polish with AI-generation, because SID_Set's
real images (sourced from OpenImages V7) are largely casual, everyday
photography rather than professional or stock imagery. This is a plausible,
specific generalization gap rather than a random failure: professional
photography is underrepresented in our "real" training distribution.

## Approaches tested and not adopted

- **PRNU-style sensor-noise features:** extracted via a median-filter
  residual, combined with CLIP. This *reduced* performance across every
  condition (e.g. clean AUC dropped from 0.970 to 0.887). We believe this
  overlaps substantially with the signal our FFT branch already captures,
  so concatenating both added redundant, noisy dimensions rather than new
  information.
- **Multi-scale FFT (sizes 16/32/64/128px, concatenated):** consistently
  underperformed our original single-scale (32px) FFT features, and
  performance degraded further as more scales were added (four-scale
  combination scored lowest of all variants tested). We interpret this as
  added feature dimensionality without added signal, given our
  training-set size.
- **Azimuthal FFT averaging** (collapsing the 2D spectrum into a 1D radial
  profile): improved center-crop robustness in isolation (crop AUC 0.9506
  vs. our un-augmented baseline's 0.8437) but performed worse than our
  final approach on blur and resize (e.g. blur σ2 AUC 0.7577 vs. 0.9239),
  since azimuthal averaging assumes roughly radial symmetry in the
  artifact pattern — an assumption blur and resize don't respect as
  cleanly as cropping does.
- **Contrastive pretraining** (NT-Xent loss on a small projection head over
  frozen CLIP embeddings, 5 epochs): achieved only 0.72 clean AUC at
  hackathon scale, well below our final model, and took roughly 400× longer
  to train (~7 minutes vs. ~1 second for our logistic regression
  classifiers) for a worse result. We believe this approach needs
  substantially more data and training time than was available to show its
  potential.

## What we'd improve with more time

- Evaluate against WildFake (multi-generator, in-the-wild images) for a
  broader domain-shift check; we were unable to load this dataset due to a
  data-integrity issue in its published metadata (a non-numeric value in a
  column the loader expects to be numeric).
- Build proper domain-adaptive threshold calibration rather than a single
  fixed cutoff.
- Train on the full SID_Set (300K images) rather than our hackathon-scale
  subsample (1,500 train / 600 validation images).
- Investigate a diffusion-reconstruction-error (DIRE-style) signal as a
  third fusion branch, which research suggests could add complementary
  information beyond CLIP and FFT.
