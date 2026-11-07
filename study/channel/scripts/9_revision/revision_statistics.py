import math
from collections import defaultdict

import numpy as np
from scipy.stats import beta, norm
from scipy.stats import t as dist_t

SEED = 20261012
N_FLIP = 100_000
N_BOOT = 10_000
EXACT_LIMIT = 20
GROUP_LIMIT = 20


def group_id(qid: str) -> str:
    return qid.rsplit("-", 1)[0]


def exact_mcnemar(a: int, b: int) -> float:
    m = a + b
    if m == 0:
        return 1.0
    k = min(a, b)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def holm(ps: list) -> list:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    output = [0.0] * len(ps)
    before = 0.0
    for pos, i in enumerate(order):
        before = max(before, min(1.0, (len(ps) - pos) * ps[i]))
        output[i] = before
    return output


def wald_interval(a: int, b: int, n: int, alpha: float = 0.05):
    z = norm.ppf(1 - alpha / 2)
    d = (a - b) / n
    var = (a + b - (a - b) ** 2 / n) / n ** 2
    se = math.sqrt(max(var, 0.0))
    return d - z * se, d + z * se


def exact_interval(a: int, b: int, n: int, alpha: float = 0.05):
    m = a + b
    if m == 0:
        return 0.0, 0.0
    lo = 0.0 if a == 0 else float(beta.ppf(alpha / 2, a, m - a + 1))
    hi = 1.0 if a == m else float(beta.ppf(1 - alpha / 2, a + 1, m - a))
    return (2 * lo - 1) * m / n, (2 * hi - 1) * m / n


def _q21_tango(a: int, b: int, n: int, d0: float) -> float:
    A = 2 * n
    B = -a - b + (2 * n - a + b) * d0
    C = -b * d0 * (1 - d0)
    return (math.sqrt(max(B * B - 4 * A * C, 0.0)) - B) / (2 * A)


def tango_interval(a: int, b: int, n: int, alpha: float = 0.05):
    z = norm.ppf(1 - alpha / 2)

    def score_statistic(d0):
        var = n * (2 * _q21_tango(a, b, n, d0) + d0 * (1 - d0))
        delta = a - b - n * d0
        if var <= 0:
            return math.copysign(math.inf, delta) if delta else 0.0
        return delta / math.sqrt(var)

    def bisect_root(lo, hi, target):
        for _ in range(100):
            mid = (lo + hi) / 2
            if score_statistic(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    d = (a - b) / n
    edge = 1e-12
    upper = bisect_root(d, 1 - edge, -z) if d < 1 - edge else 1.0
    lower = bisect_root(-1 + edge, d, z) if d > -1 + edge else -1.0
    return lower, upper


def _per_group(A: dict, B: dict, group_key=group_id):
    rows = sorted(set(A) & set(B))
    D, nb = defaultdict(int), defaultdict(int)
    for q in rows:
        g = group_key(q)
        D[g] += int(B[q]) - int(A[q])
        nb[g] += 1
    key = sorted(D)
    return rows, key, np.array([D[k] for k in key], dtype=np.int64), \
        np.array([nb[k] for k in key], dtype=np.int64)


def signflip_test(D: np.ndarray, seed: int = SEED, n_flip: int = N_FLIP):
    T = int(D.sum())
    nz = D[D != 0]
    if nz.size == 0:
        return 1.0, "enumeration", 1
    if nz.size <= EXACT_LIMIT:
        total = 1 << nz.size
        bit = np.arange(nz.size, dtype=np.int64)
        n_extreme = 0
        for start in range(0, total, 1 << 16):
            idx = np.arange(start, min(total, start + (1 << 16)), dtype=np.int64)
            mark = ((idx[:, None] >> bit) & 1) * 2 - 1
            n_extreme += int(np.sum(np.abs(mark @ nz) >= abs(T)))
        return n_extreme / total, "enumeration", total
    rng = np.random.default_rng(seed)
    n_extreme = 0
    rest = n_flip
    while rest:
        k = min(rest, 10_000)
        mark = rng.integers(0, 2, size=(k, nz.size), dtype=np.int8) * 2 - 1
        n_extreme += int(np.sum(np.abs(mark.astype(np.int64) @ nz) >= abs(T)))
        rest -= k
    return (1 + n_extreme) / (1 + n_flip), "monte_carlo", n_flip


def cluster_bootstrap_interval(D: np.ndarray, nb: np.ndarray, alpha: float = 0.05,
                      seed: int = SEED, n_boot: int = N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(D), size=(n_boot, len(D)))
    d = D[idx].sum(axis=1) / nb[idx].sum(axis=1)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cr1_wald_interval(D: np.ndarray, nb: np.ndarray, alpha: float = 0.05, t: bool = False):
    G, N = len(D), int(nb.sum())
    d = D.sum() / N
    e = D - d * nb
    var = G / (G - 1) * float(np.sum(e ** 2)) / N ** 2
    k = float(dist_t.ppf(1 - alpha / 2, G - 1)) if t else float(norm.ppf(1 - alpha / 2))
    return float(d - k * math.sqrt(var)), float(d + k * math.sqrt(var))


def _p(x: float) -> float:
    return float(f"{x:.6g}")


def _r(x: float) -> float:
    return round(float(x), 6)


def _bound(a, b, n, D, nb, alpha, seed):
    wl, wh = wald_interval(a, b, n, alpha)
    el, eh = exact_interval(a, b, n, alpha)
    tl, th = tango_interval(a, b, n, alpha)
    bl, bh = cluster_bootstrap_interval(D, nb, alpha, seed)
    cl, ch = cr1_wald_interval(D, nb, alpha)
    nz = int(np.sum(D != 0))
    if nz >= GROUP_LIMIT:
        gl, gh, method = bl, bh, "cluster bootstrap percentile"
    else:
        gl, gh = cr1_wald_interval(D, nb, alpha, t=True)
        method = f"CR1 with t({len(D) - 1}), fewer than {GROUP_LIMIT} nonzero groups"
    lo, hi = min(wl, el), max(wh, eh)
    if a + b == 0:
        lo, hi = min(lo, tl), max(hi, th)
    return {
        "row_interval": [_r(lo), _r(hi)],
        "row_wald": [_r(wl), _r(wh)], "row_exact": [_r(el), _r(eh)], "row_tango": [_r(tl), _r(th)],
        "row_tango_in_bound": a + b == 0,
        "group_interval": [_r(gl), _r(gh)], "group_interval_method": method,
        "group_bootstrap_interval": [_r(bl), _r(bh)],
        "group_cr1_interval": [_r(cl), _r(ch)],
        "largest_gain_pp": round(100 * max(hi, gh, ch), 3),
        "largest_loss_pp": round(-100 * min(lo, gl, cl), 3),
    }


def contrast(A: dict, B: dict, alpha: float = 0.05, seed: int = SEED, m: int = 1,
            clusters: dict = None) -> dict:
    rows, key, D, nb = _per_group(A, B)
    n = len(rows)
    a = sum(1 for q in rows if B[q] and not A[q])
    b = sum(1 for q in rows if A[q] and not B[q])
    p_flip, method, draws = signflip_test(D, seed)
    result = {
        "n_rows": n, "n_groups": len(key), "groups_nonzero": int(np.sum(D != 0)),
        "resolved_A": sum(int(A[q]) for q in rows), "resolved_B": sum(int(B[q]) for q in rows),
        "B_only": a, "A_only": b, "d": round((a - b) / n, 6),
        "p_mcnemar_exact": _p(exact_mcnemar(a, b)),
        "p_signflip_group": _p(p_flip), "signflip_method": method, "signflip_draws": draws,
        "signflip_note": "approximate test of a zero mean group difference",
        **_bound(a, b, n, D, nb, alpha, seed),
        "alpha": alpha,
    }
    if m > 1:
        result["family_wise"] = {"m": m, "alpha_each": round(alpha / m, 6),
                                **_bound(a, b, n, D, nb, alpha / m, seed)}
    if clusters is not None:
        _, kk, Dk, nbk = _per_group(A, B, group_key=lambda q: clusters[q.split("|", 1)[0]])
        if len(kk) >= 2:
            kl, kh = cr1_wald_interval(Dk, nbk, alpha)
            ql, qh = cluster_bootstrap_interval(Dk, nbk, alpha, seed)
            result["coarse_clusters"] = {"n_clusters": len(kk),
                                        "cr1_interval": [_r(kl), _r(kh)],
                                        "bootstrap_interval": [_r(ql), _r(qh)],
                                        "largest_gain_pp": round(100 * max(kh, qh), 3)}
        else:
            result["coarse_clusters"] = {"n_clusters": len(kk), "note": "one cluster, not computable"}
    return result


def contrast_family(result: dict, key_p: str = "p_signflip_group") -> dict:
    name = list(result)
    for kp in (key_p, "p_mcnemar_exact"):
        for n, ph in zip(name, holm([result[x][kp] for x in name])):
            result[n][kp.replace("p_", "p_holm_", 1)] = _p(ph)
    return result
