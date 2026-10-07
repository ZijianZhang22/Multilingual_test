#!/usr/bin/env python3
"""Reusable Layer-representation subspace extractors for mechanism experiments."""

from collections import defaultdict

import torch


def orthonormalize(q, rank=None):
    q = q.float()
    if q.numel() == 0:
        return q
    q, _ = torch.linalg.qr(q, mode="reduced")
    if rank is not None:
        q = q[:, : min(rank, q.shape[1])]
    return q


def orthonormal_random(dim, rank, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(dim, rank, generator=g)
    return orthonormalize(q, rank)


def _covariance(x):
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    denom = max(x.shape[0] - 1, 1)
    return (x.T @ x) / denom


def fit_isr_cov(
    x,
    languages,
    labels,
    *,
    rank=64,
    class_label=0,
    min_examples=20,
):
    """Robust multi-environment ISR-Cov via a projection/flag mean.

    The original ISR-Cov recovers invariant directions as eigenvectors of a
    class-conditional covariance difference whose eigenvalues have the smallest
    absolute magnitude. With >2 environments, the paper proposes aggregating
    pairwise recovered subspaces via a flag mean. For equal-rank subspaces, the
    leading eigenspace of the mean projection matrix is a standard projection-
    mean implementation of this aggregation.

    Here environments are languages and class_label is one fixed XNLI class.
    """
    x = x.float().cpu()
    labels = torch.as_tensor(labels, dtype=torch.long)
    envs = sorted(set(languages))
    d = x.shape[1]
    rank = min(int(rank), d)

    covs = {}
    counts = {}
    for env in envs:
        idx = torch.tensor(
            [
                i
                for i, (lang, y) in enumerate(zip(languages, labels.tolist()))
                if lang == env and y == class_label
            ],
            dtype=torch.long,
        )
        if len(idx) < min_examples:
            continue
        covs[env] = _covariance(x[idx])
        counts[env] = int(len(idx))

    usable = sorted(covs)
    if len(usable) < 2:
        raise ValueError(
            f"ISR-Cov needs >=2 environments with >= {min_examples} "
            f"examples for class {class_label}; got {counts}"
        )

    projectors = []
    pair_diagnostics = []
    for i, a in enumerate(usable):
        for b in usable[i + 1 :]:
            delta = covs[a] - covs[b]
            # Symmetric eigen-decomposition; smallest |eigenvalue| directions
            # are the candidate invariant directions in ISR-Cov.
            evals, evecs = torch.linalg.eigh(delta)
            order = torch.argsort(evals.abs())
            q = evecs[:, order[:rank]]
            q = orthonormalize(q, rank)
            projectors.append(q @ q.T)
            pair_diagnostics.append(
                {
                    "env_a": a,
                    "env_b": b,
                    "min_abs_eval": float(evals.abs().min()),
                    "median_abs_eval": float(evals.abs().median()),
                    "max_abs_eval": float(evals.abs().max()),
                }
            )

    mean_projector = torch.stack(projectors).mean(dim=0)
    pevals, pevecs = torch.linalg.eigh(mean_projector)
    q_final = pevecs[:, torch.argsort(pevals, descending=True)[:rank]]
    q_final = orthonormalize(q_final, rank)

    return q_final, {
        "algorithm": "ISR-Cov robust projection-mean aggregation",
        "class_label": int(class_label),
        "environments": usable,
        "counts": counts,
        "n_pairs": len(projectors),
        "rank": int(q_final.shape[1]),
        "mean_projector_top_eigenvalues": [
            float(v)
            for v in pevals[torch.argsort(pevals, descending=True)[: min(rank, 10)]]
        ],
        "pair_diagnostics": pair_diagnostics,
    }


def fit_isr_multiclass_semantic(
    x,
    languages,
    labels,
    *,
    rank=64,
    sv_tol=1e-6,
):
    """ISR-Multiclass spurious recovery + semantic PCA in its null space.

    Exact ISR-Multiclass:
      1) for every class, compute environment-specific class means,
      2) PCA those E means and retain E-1 environment-varying directions,
      3) stack class-specific directions,
      4) SVD to recover the common spurious subspace.

    In a 896-d hidden state with only a handful of language environments, exact
    ISR-Multiclass can recover only <= (E-1)*K spurious directions and therefore
    leaves a very high-dimensional invariant nullspace. To create a fixed-rank
    basis suitable for matched-rank causal tests, we follow exact spurious
    recovery first, project anchor features into its nullspace, then take the
    top semantic-variance PCA directions there. The returned basis is therefore
    named an ISR-Multiclass-derived invariant semantic subspace, not the full
    ISR nullspace.
    """
    x = x.float().cpu()
    labels = torch.as_tensor(labels, dtype=torch.long)
    envs = sorted(set(languages))
    classes = sorted(set(labels.tolist()))
    d = x.shape[1]

    class_bases = []
    class_diag = {}
    for y in classes:
        means = []
        used_envs = []
        for env in envs:
            idx = torch.tensor(
                [
                    i
                    for i, (lang, yy) in enumerate(zip(languages, labels.tolist()))
                    if lang == env and yy == y
                ],
                dtype=torch.long,
            )
            if len(idx) == 0:
                continue
            means.append(x[idx].mean(dim=0))
            used_envs.append(env)

        if len(means) < 2:
            continue

        m = torch.stack(means)
        m = m - m.mean(dim=0, keepdim=True)
        # Right singular vectors are directions in hidden-feature coordinates.
        _, s, vh = torch.linalg.svd(m, full_matrices=False)
        qk = min(len(used_envs) - 1, int((s > sv_tol).sum().item()))
        if qk <= 0:
            continue
        pk = vh[:qk].T
        class_bases.append(pk)
        class_diag[str(y)] = {
            "environments": used_envs,
            "rank": int(qk),
            "singular_values": [float(v) for v in s[:qk]],
        }

    if not class_bases:
        raise ValueError("ISR-Multiclass could not recover any class-specific directions.")

    total = torch.cat(class_bases, dim=1)
    u, s_total, _ = torch.linalg.svd(total, full_matrices=False)
    spu_rank = int((s_total > sv_tol).sum().item())
    q_spurious = orthonormalize(u[:, :spu_rank], spu_rank)

    # Project all features into the estimated invariant nullspace, then choose
    # a stable fixed-rank semantic basis by PCA.
    xc = x - x.mean(dim=0, keepdim=True)
    if q_spurious.numel():
        x_inv = xc - (xc @ q_spurious) @ q_spurious.T
    else:
        x_inv = xc

    q_req = min(int(rank), x_inv.shape[0], x_inv.shape[1])
    _, s_inv, v_inv = torch.pca_lowrank(x_inv, q=q_req, center=False)
    q_invariant = v_inv[:, :q_req]
    if q_spurious.numel():
        q_invariant = q_invariant - q_spurious @ (q_spurious.T @ q_invariant)
    q_invariant = orthonormalize(q_invariant, q_req)

    return q_invariant, q_spurious, {
        "algorithm": "ISR-Multiclass spurious recovery + nullspace semantic PCA",
        "environments": envs,
        "classes": classes,
        "class_specific": class_diag,
        "stacked_rank": int(total.shape[1]),
        "estimated_spurious_rank": int(spu_rank),
        "returned_invariant_semantic_rank": int(q_invariant.shape[1]),
        "spurious_singular_values": [float(v) for v in s_total[: min(20, len(s_total))]],
        "invariant_pca_singular_values": [float(v) for v in s_inv[: min(20, len(s_inv))]],
    }


def _off_diagonal(x):
    n = x.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return x[mask]


def fit_vicreg_linear(
    x,
    pair_ids,
    *,
    rank=64,
    epochs=300,
    lr=3e-2,
    sim_weight=25.0,
    var_weight=25.0,
    cov_weight=1.0,
    gamma=1.0,
    seed=0,
    device=None,
):
    """Learn an orthonormal linear cross-lingual subspace with VICReg loss.

    Rows sharing pair_id are treated as multiple language views of the same
    semantic item. The trainable projector is represented by an orthonormal
    basis Q in original hidden coordinates, z = x Q.

    A single global scalar normalizes input energy without rotating the hidden
    coordinate system, so the learned Q can be used directly for interventions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = x.float().to(device)
    rank = min(int(rank), x.shape[1])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Group all available language views of each aligned semantic item.
    groups = defaultdict(list)
    for i, pid in enumerate(pair_ids):
        groups[str(pid)].append(i)
    groups = [idx for idx in groups.values() if len(idx) >= 2]
    if not groups:
        raise ValueError("VICReg-linear needs aligned groups with >=2 language views.")

    # Keep basis in original coordinates; only apply a global scalar for
    # numerical conditioning.
    x_center = x - x.mean(dim=0, keepdim=True)
    scale = x_center.pow(2).mean().sqrt().clamp_min(1e-6)
    xn = x_center / scale

    w = torch.randn(x.shape[1], rank, device=device) / (x.shape[1] ** 0.5)
    w = torch.nn.Parameter(w)
    opt = torch.optim.Adam([w], lr=lr)

    history = []
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        q, _ = torch.linalg.qr(w, mode="reduced")
        z = xn @ q

        # Invariance: every language view approaches its semantic group mean.
        inv_terms = []
        for idxs in groups:
            zg = z[idxs]
            inv_terms.append((zg - zg.mean(dim=0, keepdim=True)).pow(2).mean())
        inv_loss = torch.stack(inv_terms).mean()

        # Variance and covariance terms follow VICReg's anti-collapse logic.
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.relu(gamma - std).pow(2).mean()

        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
        cov_loss = _off_diagonal(cov).pow(2).mean()

        loss = (
            sim_weight * inv_loss
            + var_weight * var_loss
            + cov_weight * cov_loss
        )
        loss.backward()
        opt.step()

        if epoch in {0, epochs - 1} or (epoch + 1) % max(epochs // 10, 1) == 0:
            history.append(
                {
                    "epoch": int(epoch + 1),
                    "loss": float(loss.detach().cpu()),
                    "invariance": float(inv_loss.detach().cpu()),
                    "variance": float(var_loss.detach().cpu()),
                    "covariance": float(cov_loss.detach().cpu()),
                }
            )

    with torch.no_grad():
        q, _ = torch.linalg.qr(w, mode="reduced")
        q = q[:, :rank].detach().cpu()

    return q, {
        "algorithm": "VICReg-inspired orthonormal linear projector",
        "rank": int(q.shape[1]),
        "epochs": int(epochs),
        "lr": float(lr),
        "sim_weight": float(sim_weight),
        "var_weight": float(var_weight),
        "cov_weight": float(cov_weight),
        "gamma": float(gamma),
        "global_input_scale": float(scale.detach().cpu()),
        "n_aligned_groups": int(len(groups)),
        "history": history,
    }
