import torch
import torch.nn.functional as F

def tpt_entropy_loss(logits, top_k_fraction=0.1):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    per_view_entropy = -(probs * log_probs).sum(dim=-1)
    k = max(1, int(top_k_fraction * per_view_entropy.shape[0]))
    _, indices = torch.topk(per_view_entropy, k=k, largest=False)

    avg_probs = probs[indices].mean(dim=0)
    return -(avg_probs * avg_probs.log()).sum()