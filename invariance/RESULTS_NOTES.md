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
