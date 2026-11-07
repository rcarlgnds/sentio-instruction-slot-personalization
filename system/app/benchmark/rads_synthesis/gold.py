import random
from collections import defaultdict

SPLIT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def assign_splits(queries: list, seed: int = SPLIT_SEED) -> dict:
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for q in queries:
        by_class[q.class_label].append(q)

    counts = {"train": 0, "dev": 0, "test_holdout": 0}
    for label in sorted(by_class):
        group = sorted(by_class[label], key=lambda q: q.query_id)
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * SPLIT_RATIOS[0])
        n_dev = int(n * SPLIT_RATIOS[1])
        for i, q in enumerate(group):
            if i < n_train:
                q.split = "train"
            elif i < n_train + n_dev:
                q.split = "dev"
            else:
                q.split = "test_holdout"
            counts[q.split] += 1
    return counts
