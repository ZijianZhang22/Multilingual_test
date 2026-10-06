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
