from math import comb


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(ps: list) -> list:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, highest = [0.0] * len(ps), 0.0
    for pos, i in enumerate(order):
        highest = max(highest, min(1.0, (len(ps) - pos) * ps[i]))
        out[i] = highest
    return out
