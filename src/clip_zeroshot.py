import torch
from tqdm.notebook import tqdm

def build_and_cache_text_features(model, tokenizer, classnames, templates, device, cache_dir, file_name):
    with torch.no_grad():
        text_features = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).cpu()
        saved_path = f"{cache_dir}/{file_name}.pt"

    torch.save(text_features.cpu(), saved_path)
    print(f"Text features has been saved at {saved_path}")

    return text_features

    


def build_and_cache_image_features(model, device, dataloader, cache_dir, file_name):
    image_features = []
    image_labels = []

    for images, labels in tqdm(dataloader):
        images = images.to(device)

        with torch.no_grad():
            feats = model.encode_image(images)
            feats /= feats.norm(dim=-1, keepdim=True)

            image_features.append(feats.cpu().float())
            image_labels.append(labels)

    image_features = torch.cat(image_features)
    image_labels = torch.cat(image_labels)
    saved_path = f"{cache_dir}/{file_name}.pt"

    features = {"image_features": image_features, "labels": image_labels}

    torch.save(features, saved_path)
    print(f"Image features and labels has been saved at {saved_path}")

    return features


def top_k_accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def load_cached_features(image_path, text_path):
    cached = torch.load(image_path)
    image_features = cached["image_features"]
    labels = cached["labels"]
    text_features = torch.load(text_path)

    return {"image_features": image_features, "labels": labels, "text_features": text_features}

def load_cached_image_features(path):
    cached = torch.load(path)
    image_features = cached["image_features"]
    labels = cached["labels"]
    return {"image_features": image_features, "labels": labels}

def load_cached_text_features(path):
    text_features = torch.load(path)
    return {"text_features": text_features}