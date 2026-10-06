# Route A — Provisional Results Log

## 2026-10-05: Layer-12 representation benchmark

Model: Qwen2.5-0.5B  
Task: XNLI (EN / ZH / FR)  
Candidate layer: 12  
Representation benchmark methods: ERM, IRM, V-REx, DANN, INLP

Observed first-run results:

| Method | All-language XNLI acc | Language-ID acc | Mean LOO acc |
|---|---:|---:|---:|
| ERM | 0.5622 | 0.6114 | ~0.494 |
| IRM | 0.5639 | 0.7750 | ~0.481 |
| V-REx | 0.5656 | 0.6406 | ~0.505 |
| DANN | 0.5825 | 0.9975 | ~0.528 |
| INLP | 0.5578 | 0.9778 | ~0.536 |

LOO examples:
- ERM: EN 0.5167, FR 0.4742, ZH 0.4917
- IRM: EN 0.4833, FR 0.4833, ZH 0.4758
- V-REx: EN 0.5250, FR 0.4858, ZH 0.5033
- DANN: EN 0.5458, FR 0.5225, ZH 0.5167
- INLP: EN 0.5258, FR 0.5408, ZH 0.5417

### Working interpretation

1. Layer 12 remains the strongest candidate shared / transferable multilingual region.
2. Better leave-one-language-out transfer is **not equivalent** to lower linear language decodability.
3. DANN and INLP improved held-out transfer in this run even though language ID remained almost perfectly linearly decodable.
4. Therefore a useful multilingual representation may preserve both:
   - a shared task-relevant component, and
   - a language-specific component.
5. The stronger research question may be:
   **What representation properties enable cross-lingual transfer, and which components predict later forgetting?**
   This is more nuanced than assuming that "more language-invariant = better transfer."

### Important caveat discovered after the first run

The first version of the benchmark used method-specific random seed offsets. That means method comparisons were partially confounded by different projection/head initializations.

This has been fixed in commit:
- a52f014 Fix paired seeding in representation benchmark

All methods at the same layer now use the same initialization seed. The benchmark should be re-run before treating method differences as reliable.

### Immediate next experiments

1. Re-run the layer-12 benchmark with paired seeds.
2. Repeat with multiple experiment seeds (recommended: 0, 1, 2).
3. Only after confirming the pattern:
   - sweep INLP removal strength / iterations;
   - sweep DANN adversarial strength;
   - add RAW/PCA/random-projection controls;
   - then move to continual-language checkpoints and test which representation change best predicts forgetting.


## 2026-10-05: Paired-seed rerun confirms the main pattern

After fixing method-specific seed offsets, the layer-12 benchmark was rerun with paired initialization.

| Method | All-language XNLI acc | Language-ID acc | LOO EN | LOO FR | LOO ZH | Mean LOO |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.5617 | 0.6653 | 0.5133 | 0.4683 | 0.4883 | 0.4900 |
| IRM | 0.5675 | 0.8950 | 0.4817 | 0.5042 | 0.3942 | 0.4600 |
| V-REx | 0.5669 | 0.6078 | 0.5275 | 0.4717 | 0.4808 | 0.4933 |
| DANN | 0.5800 | 0.9983 | 0.5358 | 0.5350 | 0.5342 | 0.5350 |
| INLP | 0.5511 | 0.9692 | 0.5242 | 0.5183 | 0.5525 | 0.5317 |

### Updated interpretation

The earlier qualitative pattern survives the paired-seed rerun:

- DANN and INLP still give the strongest held-out-language transfer.
- IRM does not improve held-out transfer overall and substantially hurts held-out ZH in this run.
- V-REx is close to ERM.
- Crucially, the methods with the best LOO transfer (DANN / INLP) still retain extremely high linearly decodable language identity.

This strengthens the hypothesis that:
**cross-lingual transferability is not equivalent to low language decodability.**

A plausible representation picture is not "language-neutral only", but coexistence of:
- transferable/shared task structure, and
- language-specific structure.

### Next checks before continual-forgetting experiments

1. Repeat paired benchmark for seeds 0/1/2.
2. Sweep INLP removal strength (8/16/32/64 iterations).
3. Improve/sweep DANN adversarial training because the current DANN representation remains ~99.8% linearly language-decodable.
4. Add RAW/PCA/random-projection controls.
5. Only after the above are stable, use the best/most-informative representation metrics to predict forgetting across sequential language adaptation.


## 2026-10-05: Full 3-seed transfer/invariance tradeoff sweep

Layer 12, seeds 0/1/2, XNLI EN/ZH/FR.

Aggregate results:

| Config | Mean LOO | Task acc | Language-ID acc |
|---|---:|---:|---:|
| raw | 0.5426 ± 0.0038 | 0.5533 ± 0.0009 | 0.9994 ± 0.0000 |
| INLP-32 | 0.5423 ± 0.0015 | 0.5494 ± 0.0032 | 0.7169 ± 0.0126 |
| INLP-16 | 0.5373 ± 0.0009 | 0.5476 ± 0.0019 | 0.9225 ± 0.0090 |
| INLP-8 | 0.5371 ± 0.0049 | 0.5535 ± 0.0022 | 0.9715 ± 0.0017 |
| INLP-64 | 0.5348 ± 0.0042 | 0.5426 ± 0.0058 | 0.5641 ± 0.0058 |
| DANN λ=1 | 0.5327 ± 0.0031 | 0.5802 ± 0.0011 | 0.9981 ± 0.0002 |
| ERM | 0.4940 ± 0.0062 | 0.5617 ± 0.0020 | 0.6135 ± 0.0378 |
| V-REx | 0.4922 ± 0.0100 | 0.5569 ± 0.0175 | 0.7495 ± 0.1126 |
| IRM | 0.4854 ± 0.0199 | 0.5644 ± 0.0027 | 0.7334 ± 0.1181 |
| PCA-64 | 0.4600 ± 0.0080 | 0.4769 ± 0.0074 | 0.9997 ± 0.0000 |
| RandomProj-64 | 0.3733 ± 0.0139 | 0.3677 ± 0.0133 | 0.9848 ± 0.0049 |

### Key observations

1. Raw layer-12 features are already the strongest LOO baseline (0.5426), confirming that Qwen's pretrained middle-layer representation already contains strong transferable structure.
2. INLP-32 is statistically/practically tied with raw LOO (0.5423 vs 0.5426) while reducing linear language-ID accuracy from ~99.94% to ~71.69%.
3. Stronger erasure (INLP-64) lowers language-ID further (~56.4%) but begins to reduce task accuracy and LOO transfer, suggesting over-erasure can remove useful transferable information.
4. Therefore lower language decodability is not monotonically associated with better transfer. A substantial amount of linearly decodable language identity can be removed with almost no loss of cross-lingual transfer, but removing too much starts to hurt.
5. DANN λ=1 gives the best all-language task accuracy (~58.0%) but leaves language ID almost perfectly decodable and does not beat raw LOO. Its current benefit appears to be task regularization rather than successful language invariance.
6. Learned 64-d ERM/IRM/V-REx projections lower some language decodability but lose substantial LOO performance relative to raw, so dimensional compression / task fitting alone is not a sufficient explanation for transfer.

### Stronger working hypothesis

Layer-12 representations appear to contain partly separable components:
- a transferable/shared task component;
- a highly decodable language-specific component.

The INLP sweep suggests a useful mechanistic decomposition: approximately 32 rounds of language-direction removal preserve nearly all held-out-language transfer while substantially reducing language identity, whereas stronger removal starts damaging transferable/task information.

This motivates using the INLP removed basis and its residual complement as candidate language-specific vs transferable/shared subspaces in continual-language adaptation. Track each subspace separately across checkpoints and test which drift predicts forgetting.


## 2026-10-06: Corrected anchor-relative subspace tracking pilot

The seed bug in checkpoint-specific INLP refits was fixed, so identical stage-0 checkpoints now have language-subspace overlap 1.0 in both sequence branches.

Behavioral forgetting:
- EN→ZH: EN loss 2.5191 after EN training, then 2.5347 after ZH; forgetting = +0.015650.
- ZH→EN: ZH loss 2.6467 after ZH training, then 2.6649 after EN; forgetting = +0.018215.

Anchor-relative Layer-12 drift for the forgotten language:
- EN forgotten by ZH: language drift L2 = 0.323947; shared-complement drift L2 = 1.207635; anchor language-subspace overlap = 0.393069.
- ZH forgotten by EN: language drift L2 = 0.242556; shared-complement drift L2 = 1.467532; anchor language-subspace overlap = 0.410887.

The higher-forgetting case has larger shared-complement drift despite smaller language-subspace drift. This is only a two-point directional observation, not statistical evidence.

Dimension-normalized squared drift is also informative because the candidate language subspace is 64-D and its complement is 832-D:
- EN forgotten by ZH: language ≈ 0.001640 per dim; shared ≈ 0.001753 per dim (ratio ≈ 0.94).
- ZH forgotten by EN: language ≈ 0.000919 per dim; shared ≈ 0.002589 per dim (ratio ≈ 0.36).

Thus the second forgetting event shows substantially more shared-complement perturbation per dimension.

Important interpretation:
- The global "all rows" correlations are not evidence about forgetting because they mix trained-language improvement, zero-shot languages, and old-language forgetting.
- Only old-language rows answer the forgetting question, and n=2 is far too small for correlation analysis.
- This pilot is therefore used only to motivate a causal intervention experiment: preserve different Layer-12 components during the second-language stage and compare forgetting directly.
