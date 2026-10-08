# Semantic Causal Validation

This directory adds a focused validation stage for the multilingual retention project. It does not overwrite the existing Step1-7 outputs.

## Scientific questions

1. Pooling mismatch: earlier behavioral interventions edited the last prompt token while several bases were fitted on mean-pooled states. We refit the core subspaces using last-token features and compare mean-vs-last geometry.
2. Semantic specificity: we measure XNLI task-probe accuracy, language-ID probe accuracy, cross-lingual retrieval of the same aligned XNLI item, and same-item cross-language cosine margins.
3. Magnitude-controlled causal specificity: the feature intervention sweep compares same-rank candidates after scaling every perturbation down to a common energy.
4. Feature-matched Drift/ISR rescue: Step7 is rerun with last-token-fitted bases, eliminating the mean-pool / last-token mismatch.

## Files

- analyze_semantic_specificity.py: task vs language probe selectivity, cross-lingual semantic-item retrieval, aligned semantic cosine margins.
- run_intervention_strength_sweep.py: natural beta sweep, rank/equal-energy beta sweep, frozen XNLI and language-ID probes.
- compare_pooling_subspaces.py: principal-angle overlap between mean-pooled and last-token bases.
- run_semantic_validation_suite.sh: one-command runner.

## Recommended first run: Qwen2.5-7B

Run from the repository root:

    cd /workspace/Multilingual_test
    git checkout zijian_test
    git pull

    export ANCHOR_CHECKPOINT=/workspace/Multilingual_test/replication_runs/qwen25_7b_a100_fresh_fullzh/seed0/anchor
    export ADAPTED_CHECKPOINT=/workspace/Multilingual_test/replication_runs/qwen25_7b_a100_fresh_fullzh/seed0/lr_4e-05/adapted
    export LAYER=23
    export MODEL_TAG=qwen25_7b_semantic_seed0
    export EVAL_BATCH=2
    export EXTRACT_BATCH=2

    nohup bash experiments/semantic_causal_validation/run_semantic_validation_suite.sh > semantic_validation_7b.log 2>&1 &
    tail -f semantic_validation_7b.log

If the old mean-pooled core-subspace file is still available, set MEAN_CORE=/path/to/legacy/core_subspaces.pt before launching. The suite will also quantify how much the fitted subspaces change when switching from mean pooling to last-token pooling.

## Primary outputs

mechanism_runs/semantic_causal_validation/<MODEL_TAG>/ contains:
- last_pool_subspaces/core_subspaces.pt and summary.json
- last_pool_subspaces/features/adapted_probe.pt and adapted_aligned.pt
- last_pool_drift_isr_partition/partition_rescue_summary.csv
- last_pool_drift_isr_partition/principal_alignment_spectrum.csv
- semantic_specificity/semantic_specificity_summary.csv
- semantic_specificity/retrieval_by_language_pair.csv
- feature_intervention_sweep/intervention_strength_sweep.csv
- feature_intervention_sweep/rank32_primary_equal_energy.csv
- pooling_comparison/pooling_subspace_overlap.csv (optional)
- semantic_causal_validation_results.tar.gz

## What would count as strong evidence?

A direction should not be called semantic from probe accuracy alone. A stronger pattern is: task information above random controls, language ID near chance or weaker than task information, high cross-lingual same-item retrieval, positive same-item semantic cosine margin, causal task degradation at equal intervention energy, and replication when the basis and intervention both use last-token states.

For the current Drift/ISR partition, the primary comparison is top32 vs bottom32 vs random32 draws under the rank_energy_matched rows of rank32_primary_equal_energy.csv.

If bottom32 remains more disruptive to NLI than top32 and random at equal energy, the evidence supports behavioral/semantic-task density outside the ISR-aligned component. If language-ID effects dominate instead, the directions should not be described as purely semantic.

## Interpretation boundary

These tests can support that a subspace is causally relevant to semantic-task behavior, but they do not by themselves identify a human-interpretable semantic concept. Strong semantic identification would require consistent label- or relation-specific steering/swap effects on a much larger example set.
