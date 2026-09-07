import torch

def tip_adapter_logits(device, test_features, cache_keys, cache_values, text_features, logit_scale, alpha=1.5, beta=5.0):
    sim = test_features.to(device) @ cache_keys.T.to(device)
    A = torch.exp(-beta * (1 - sim)).to(device)
    cache_logits = A @ cache_values.to(device)
    zero_shot_logits = logit_scale * (test_features.to(device) @ text_features.to(device))
    return zero_shot_logits + alpha * cache_logits