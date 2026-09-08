import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
import random

def load_imagenet_sketch(root, preprocess, imagenet_class_names, k_shot=16, seed=42):
    """Returns the standard harness bundle for ImageNet-Sketch."""
    random.seed(seed)

    dataset = ImageFolder(root=root, transform=preprocess)
    raw_dataset = ImageFolder(root=root) 

    # group indices by class for few-shot sampling
    from collections import defaultdict
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples): 
        class_to_indices[label].append(idx)

    few_shot_indices = []
    for label, indices in class_to_indices.items():
        few_shot_indices.extend(random.sample(indices, min(k_shot, len(indices))))

    eval_indices = list(set(range(len(dataset))) - set(few_shot_indices))

    return {
        "class_names": imagenet_class_names,          # 1000 names, aligned to labels
        "few_shot_ds": Subset(dataset, few_shot_indices),
        "eval_ds": Subset(dataset, eval_indices),
        "num_classes": 1000,
        "raw_dataset": raw_dataset,
        "eval_indices": eval_indices,
    }