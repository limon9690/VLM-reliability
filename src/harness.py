import torch
from tpt import tpt_entropy_loss
import torchvision.transforms as transforms
from tqdm.notebook import tqdm


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


def run_tpt(model, prompt_learner, text_encoder, preprocess, raw_images, true_labels, device,
            augment_transform, metrics, n_views=63, lr=0.005, top_k=0.1, subset=200):
    all_logits = []
    all_labels = []

    for i in tqdm(range(min(subset, len(raw_images)))):
        image = raw_images[i]
        label = true_labels[i]

        prompt_learner.reset_context()
        optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=lr)


        views = generate_N_views(n_views, augment_transform, preprocess, image).to(device)
        img_feats = model.encode_image(views)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        prompts, tok = prompt_learner()
        txt = text_encoder(prompts, tok)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        logits = img_feats @ txt.t()
        loss = tpt_entropy_loss(logits, top_k)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

 
        with torch.no_grad():
            clean = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            cf = model.encode_image(clean); cf = cf / cf.norm(dim=-1, keepdim=True)
            prompts, tok = prompt_learner()
            txt = text_encoder(prompts, tok); txt = txt / txt.norm(dim=-1, keepdim=True)
            clean_logits = model.logit_scale.exp() * (cf @ txt.t())        

        all_logits.append(clean_logits.cpu())
        all_labels.append(label)

        del views, img_feats, logits, loss, txt, prompts, clean_logits
        torch.cuda.empty_cache()


    all_logits = torch.cat(all_logits, dim=0)    
    all_labels = torch.tensor(all_labels)               

    result = {name: fn(all_logits, all_labels) for name, fn in metrics.items()}
    result["n"] = len(all_labels)

    probs = all_logits.softmax(dim=-1)
    confidences, _ = probs.max(dim=-1)
    print("TPT mean confidence:", confidences.mean().item())
    print("TPT min/max confidence:", confidences.min().item(), confidences.max().item())

    return result

def run_comparison(shared, methods, metrics):
    results = {}
    for name, spec in methods.items():
        logits = spec["fn"](**shared, **spec["params"])
        results[name] = evaluate(logits, shared["labels"], metrics)
    return results


def generate_N_views(N, transform_fn, preprocess, image):
    views = [transform_fn(image) for _ in range(N)]
    clean = preprocess(image)   
    views.append(clean)
    return torch.stack(views)


def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return 100 * (preds == labels).float().mean().item()


def ece(logits, labels, n_bins=10):
    probs = logits.softmax(dim=-1)
    confidences, preds = probs.max(dim=-1)
    accuracies = (preds == labels).float()

    ece_val = 0.0
    bin_edges = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        prop = in_bin.float().mean()

        if prop > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece_val += (bin_acc - bin_conf).abs() * prop

    return ece_val.item() * 100


def signed_gap(logits, labels):
    probs = logits.softmax(dim=-1)
    conf, pred = probs.max(dim=-1)
    acc = (pred == labels).float().mean().item()
    return conf.mean().item() - acc   # positive = overconfident