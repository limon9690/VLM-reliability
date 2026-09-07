import torch


def evaluate(logits, labels, metrics):
    return {name: fn(logits, labels) for name, fn in metrics.items()}


def zero_shot_logits(test_features, text_features, logit_scale, **cfg):
    return logit_scale * (test_features @ text_features)


def tip_adapter_logits(test_features, text_features, logit_scale, cache_keys, cache_values, alpha, beta, **cfg):
    sim = test_features @ cache_keys.T
    A = torch.exp(-beta * (1 - sim))
    cache_logits = A @ cache_values
    zs_logits = logit_scale * (test_features @ text_features)
    return zs_logits + alpha * cache_logits


def coop_logits(test_features, coop_text_features, logit_scale, **cfg):
    return logit_scale * (test_features @ coop_text_features.t())


def run_comparison(shared, methods, metrics):
    results = {}
    for name, spec in methods.items():
        logits = spec["fn"](**shared, **spec["params"])
        results[name] = evaluate(logits, shared["labels"], metrics)
    return results

