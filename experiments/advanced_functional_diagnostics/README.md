# Advanced Functional Diagnostics

This directory is intentionally separate from:

`experiments/retention_subspace_mechanism/`

and writes to:

`mechanism_runs/advanced_functional_diagnostics_v1/`

by default. It is designed to extend the multilingual forgetting project without
overwriting the existing Step 1-5 mechanism pipeline.

## What it measures

### A. Spectral / effective-rank drift

`measure_spectral_output_drift.py`

For every requested layer:

- relative hidden drift RMS
- participation-ratio effective dimension
- entropy effective rank
- stable rank
- rank needed for 50/80/90/95/99% drift energy
- top-k drift-energy concentration

It also measures functional output drift:

- old/new LM loss
- anchor -> adapted KL
- adapted -> anchor KL
- symmetric KL
- top-1 token agreement

This distinguishes large but low-rank drift from diffuse drift, and hidden drift
from actual output/function drift.

### B. Old/new activation-gradient subspaces

`measure_gradient_subspaces.py`

For each layer it forms covariance/Gram matrices of:

- dL_old / dh_l
- dL_new / dh_l

and reports:

- principal-angle/projection overlap
- new-gradient energy inside old-gradient subspace
- old-gradient energy inside new-gradient subspace
- effective dimension / spectrum

This is more informative than a single mean gradient cosine.

### C. Alternative concept-erasure controls

`compare_concept_erasure.py`

Fits language erasers on anchor XNLI features:

- mean projection
- LEACE

and evaluates:

- fresh language probe accuracy after erasure
- XNLI task probe accuracy after erasure
- RMS feature distortion
- tokenwise causal LM-loss change at the selected layer

The causal hook uses a low-rank factorization of the erasure map for speed.

### D. Targeted improvement experiments

`run_targeted_interventions.py`

Two intervention families:

#### Projected hidden preservation

[
L = L_{new} +
lambda
rac{|(h-h_{anchor})Q|_F^2}
     {|h_{anchor}Q|_F^2+epsilon}
]

Only the selected subspace is protected.

#### Activation-gradient shielding

[
g_h' = g_h - alpha (g_h Q)Q^T
]

Only backpropagated activation gradients entering the selected subspace are
attenuated.

By default it tests:

- transfer
- isr_cov
- isr_multiclass
- vicreg
- drift

using the existing Layer-20 Step-2 artifact.

## One-click run

The main mechanism pipeline should first have produced the anchor, full-FT
adapted checkpoint, and Step-2 subspace artifact.

Then run:

```bash
cd /workspace/Multilingual_test
git checkout zijian_test
git pull

nohup bash experiments/advanced_functional_diagnostics/run_advanced_suite.sh \
  > advanced_diagnostics.log 2>&1 &

echo $!
tail -f advanced_diagnostics.log
```

The default output root is:

`mechanism_runs/advanced_functional_diagnostics_v1/`

and the compact package is:

`advanced_functional_diagnostics_v1.tar.gz`

## Scientific interpretation

The intended chain is:

measurement
-> low-rank / functional localization
-> causal validation
-> targeted intervention

The suite is meant to test whether multilingual forgetting is better predicted
by the direction and functional sensitivity of representational change than by
total drift magnitude alone.
