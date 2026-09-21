import collections
import random

def split_indices(labels, seed, n_cache=16, n_val=10):
    rng = random.Random(seed)
    by_class = collections.defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)
    cache, val, test = [], [], []
    for idx in by_class.values():
        idx = idx[:]
        rng.shuffle(idx)
        cache += idx[:n_cache]
        val += idx[n_cache:n_cache + n_val]
        test += idx[n_cache + n_val:]
    return cache, val, test