import numpy as np

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 42


def bootstrap_ci(values, n_iter: int = BOOTSTRAP_ITERS, alpha: float = 0.05,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_iter, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(arr.size)}


def paired_bootstrap_diff(a, b, n_iter: int = BOOTSTRAP_ITERS,
                          seed: int = BOOTSTRAP_SEED) -> dict:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"diff": None, "ci_low": None, "ci_high": None, "p_value": None, "n": 0}
    da = np.asarray([x for x, _ in pairs], dtype=np.float64)
    db = np.asarray([y for _, y in pairs], dtype=np.float64)
    diff = da - db
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_iter, diff.size))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    observed = float(diff.mean())
    crossings = np.sum(boot <= 0) if observed >= 0 else np.sum(boot >= 0)
    p_asl = min(2.0 * (crossings + 1) / (n_iter + 1), 1.0)
    return {"diff": observed, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(p_asl), "p_permutation": permutation_p(da, db, seed=seed),
            "n": int(diff.size)}


def permutation_p(a, b, n_iter: int = BOOTSTRAP_ITERS, seed: int = BOOTSTRAP_SEED) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.size == 0:
        return None
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_iter, diff.size))
    perm_means = np.abs((signs * diff).mean(axis=1))
    return float((1 + np.sum(perm_means >= observed)) / (1 + n_iter))


def holm_correction(pvalues: dict, alpha: float = 0.05) -> dict:
    items = sorted(((k, v) for k, v in pvalues.items() if v is not None),
                   key=lambda kv: kv[1])
    m = len(items)
    out = {}
    max_adj = 0.0
    for i, (name, p) in enumerate(items):
        adj = min((m - i) * p, 1.0)
        max_adj = max(max_adj, adj)
        out[name] = {"p": p, "p_adj": max_adj, "reject": max_adj <= alpha}
    for k, v in pvalues.items():
        if v is None:
            out[k] = {"p": None, "p_adj": None, "reject": False}
    return out
