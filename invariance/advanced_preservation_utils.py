import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_blocks(path):
    return torch.load(path, map_location="cpu")["input_ids"].long()


def make_loader(blocks, batch_size):
    return DataLoader(
        TensorDataset(blocks),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


def load_model(checkpoint, device, use_bf16, trainable=True):
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def load_tokenizer(checkpoint):
    tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@torch.no_grad()
def evaluate(model, blocks, batch_size, device, use_bf16):
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for (x,) in make_loader(blocks, batch_size):
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(input_ids=x, labels=x, use_cache=False)
        n = x.shape[0] * (x.shape[1] - 1)
        total_loss += float(out.loss) * n
        total_targets += n
    model.train()
    return total_loss / max(total_targets, 1)


def shuffled_blocks(blocks, seed, max_blocks=None):
    g = torch.Generator().manual_seed(seed)
    x = blocks[torch.randperm(len(blocks), generator=g)]
    if max_blocks is not None:
        x = x[:max_blocks]
    return x


def estimate_hidden_gradient_importance(
    checkpoint,
    blocks,
    *,
    layer,
    batch_size,
    max_batches,
    device,
    use_bf16,
):
    """Diagonal Fisher-like importance in hidden coordinates."""
    model = load_model(checkpoint, device, use_bf16, trainable=True)
    model.eval()
    total = None
    count = 0

    for batch_idx, (x,) in enumerate(make_loader(blocks, batch_size)):
        if batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(
                input_ids=x,
                labels=x,
                output_hidden_states=True,
                use_cache=False,
            )
        h = out.hidden_states[layer]
        grad_h = torch.autograd.grad(out.loss, h, retain_graph=False)[0]
        batch_imp = grad_h.float().pow(2).sum(dim=(0, 1)).detach().cpu()
        total = batch_imp if total is None else total + batch_imp
        count += grad_h.shape[0] * grad_h.shape[1]

    del model
    torch.cuda.empty_cache()
    if total is None:
        raise RuntimeError("No batches used for importance estimation")
    return total / max(count, 1)


def make_stability_weights(old_imp, new_imp, mode="ratio", eps=1e-12):
    old_imp = old_imp.float().clamp_min(0)
    new_imp = new_imp.float().clamp_min(0)

    if mode == "ratio":
        w = old_imp / (new_imp + eps)
    elif mode == "old_fraction":
        w = old_imp / (old_imp + new_imp + eps)
    elif mode == "old_only":
        w = old_imp.clone()
    else:
        raise ValueError(mode)

    positive = w[w > 0]
    if positive.numel() > 0:
        cap = torch.quantile(positive, 0.99)
        w = w.clamp_max(cap)
    w = w / w.mean().clamp_min(eps)
    return w


def fit_hidden_gradient_subspace(
    checkpoint,
    blocks,
    *,
    layer,
    rank,
    batch_size,
    max_batches,
    device,
    use_bf16,
):
    """Top eigenvectors of the old-language hidden-gradient covariance."""
    model = load_model(checkpoint, device, use_bf16, trainable=True)
    model.eval()
    cov = None
    n_rows = 0

    for batch_idx, (x,) in enumerate(make_loader(blocks, batch_size)):
        if batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(
                input_ids=x,
                labels=x,
                output_hidden_states=True,
                use_cache=False,
            )
        h = out.hidden_states[layer]
        grad_h = torch.autograd.grad(out.loss, h, retain_graph=False)[0]
        g = grad_h.float().reshape(-1, grad_h.shape[-1])
        batch_cov = g.T @ g
        cov = batch_cov if cov is None else cov + batch_cov
        n_rows += g.shape[0]

    del model
    torch.cuda.empty_cache()
    if cov is None:
        raise RuntimeError("No batches used to fit gradient subspace")

    cov = cov / max(n_rows, 1)
    evals, evecs = torch.linalg.eigh(cov)
    rank = min(rank, evecs.shape[1])
    q = evecs[:, -rank:]
    return q.detach().cpu(), evals[-rank:].detach().cpu()


def hidden_gradient_projection_hook(q, strength=1.0):
    q = q.float()

    def hook(grad):
        g = grad.float()
        projected = (g @ q) @ q.T
        out = g - strength * projected
        return out.to(grad.dtype)

    return hook


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def load_aligned_pairs(path, old_language, new_language):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], {})[row["language"]] = row

    pairs = []
    for pair_id, items in grouped.items():
        if old_language not in items or new_language not in items:
            continue
        old = items[old_language]
        new = items[new_language]
        pairs.append(
            (
                old["premise"] + "\n\n" + old["hypothesis"],
                new["premise"] + "\n\n" + new["hypothesis"],
            )
        )
    if not pairs:
        raise ValueError(
            f"No aligned {old_language}/{new_language} pairs found in {path}"
        )
    return pairs


def batch_pairs(pairs, batch_size, seed):
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        batches.append([pairs[i] for i in idxs])
    return batches


def pooled_hidden_from_texts(
    model,
    tokenizer,
    texts,
    *,
    layer,
    max_length,
    device,
    use_bf16,
    no_grad=False,
):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )
        pooled = mean_pool(out.hidden_states[layer], enc["attention_mask"])
    return pooled


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


@torch.no_grad()
def sample_hidden_tokens(
    model,
    blocks,
    *,
    layer,
    batch_size,
    max_batches,
    tokens_per_batch,
    device,
    use_bf16,
    seed,
):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    samples = []
    model.eval()
    for batch_idx, (x,) in enumerate(make_loader(blocks, batch_size)):
        if batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and use_bf16),
        ):
            out = model(
                input_ids=x,
                output_hidden_states=True,
                use_cache=False,
            )
        h = out.hidden_states[layer].float().reshape(-1, out.hidden_states[layer].shape[-1])
        n = min(tokens_per_batch, h.shape[0])
        idx = torch.randperm(h.shape[0], generator=rng)[:n].to(device)
        samples.append(h[idx].cpu())
    if not samples:
        raise RuntimeError("No SAE activation samples collected")
    return torch.cat(samples, dim=0)


def train_sparse_autoencoder(
    activations,
    *,
    dict_size,
    epochs,
    batch_size,
    lr,
    l1_lambda,
    device,
    seed,
):
    set_seed(seed)
    d_model = activations.shape[1]
    sae = SparseAutoencoder(d_model, dict_size).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)

    ds = TensorDataset(activations.float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_rec = 0.0
        total_l1 = 0.0
        n = 0
        for (x,) in loader:
            x = x.to(device)
            recon, z = sae(x)
            rec = (recon - x).pow(2).mean()
            l1 = z.abs().mean()
            loss = rec + l1_lambda * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_rec += float(rec.detach())
            total_l1 += float(l1.detach())
            n += 1
        print(
            f"[SAE epoch {epoch+1}/{epochs}] "
            f"recon={total_rec/max(n,1):.6f} l1={total_l1/max(n,1):.6f}"
        )
    return sae


@torch.no_grad()
def mean_sae_activity(
    sae,
    activations,
    *,
    batch_size,
    device,
):
    total = None
    count = 0
    loader = DataLoader(TensorDataset(activations.float()), batch_size=batch_size)
    sae.eval()
    for (x,) in loader:
        x = x.to(device)
        z = sae.encode(x).float()
        s = z.abs().sum(dim=0).cpu()
        total = s if total is None else total + s
        count += z.shape[0]
    return total / max(count, 1)
