# Retention-Subspace Mechanism Pipeline

This folder implements the next four experiments after the normalized layer-preservation pilot.

## Scientific questions

1. **Layer/strength robustness.** Is the Layer-20 retention/plasticity advantage stable across normalized preservation strengths, rather than an accident at lambda=20?
2. **Mechanism localization.** Which Layer-20 directions contain language identity, cross-language transferable semantics, and actual anchor-to-adapted drift?
3. **Causal necessity.** Does removing a real subspace hurt performance more than removing an equally ranked random subspace?
4. **Causal rescue.** Can restoring only the anchor-aligned component inside a candidate subspace recover old-language performance without erasing new-language gains?

## Candidate Layer-20 subspaces

- **language**: INLP language-decodable directions fit on anchor XNLI features.
- **transfer**: cross-language semantic PCA after removing the language subspace.
- **drift**: top PCA directions of full-FT minus anchor features on the same fixed probe.
- **random_***: independent isotropic orthonormal bases matched to each real subspace rank.

## Interventions

Removal uses centered projection:

`h' = h - beta * P_S(h - mu_anchor)`

Rescue uses exact per-token anchor/adapted differences:

`h'_adapt = h_adapt + alpha * P_S(h_anchor - h_adapt)`

The matched-rank random controls are essential. A result is interesting only when the real subspace is stronger than its random control.

## One-command run

```bash
bash experiments/retention_subspace_mechanism/run_first_four_steps.sh
```

The runner is resumable and bootstraps EN/ZH data and the EN stage-1 anchor if missing.

Default experiment:

- Qwen2.5-0.5B
- EN -> ZH
- 20% of the ZH training blocks for Step 1
- Layer 12/20/24
- normalized lambda = 5, 20, 50
- Layer-20 language/transfer/drift subspaces
- removal beta = 0.25, 0.5, 1.0
- rescue alpha = 0.25, 0.5, 1.0
- 128 validation blocks per language

The final compact package is:

`retention_subspace_first_four_steps.tar.gz`

## Interpretation

The desired pattern is not merely lower forgetting under regularization. Strong evidence would be:

- Layer 20 remains near the best retention/plasticity frontier across multiple normalized lambdas.
- A real Layer-20 subspace shows larger old-language causal effects than a same-rank random subspace.
- Anchor-direction rescue in that subspace improves old-language loss while causing little new-language loss increase.
- Different subspaces separate retention-critical and plasticity-critical representational directions.

That would support the mechanism claim that multilingual forgetting depends on **where and in which representational directions adaptation occurs**, not just the total amount of model change.
