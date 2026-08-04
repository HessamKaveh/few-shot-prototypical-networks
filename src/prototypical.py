import torch
import torch.nn.functional as F


def compute_prototypes(support_embeddings, support_labels, n_way):
    """پروتوتایپ هر کلاس = میانگین embedding نمونه‌های support همان کلاس"""
    prototypes = torch.zeros(n_way, support_embeddings.size(1), device=support_embeddings.device)
    for c in range(n_way):
        class_mask = (support_labels == c)
        prototypes[c] = support_embeddings[class_mask].mean(dim=0)
    return prototypes


def prototypical_loss(support_emb, support_labels, query_emb, query_labels, n_way):
    """
    محاسبه loss بر اساس فاصله اقلیدسی هر نمونه query تا پروتوتایپ هر کلاس.
    کلاس با کمترین فاصله = بیشترین احتمال (منفی فاصله به عنوان logit).
    """
    prototypes = compute_prototypes(support_emb, support_labels, n_way)

    # فاصله اقلیدسی بین هر query و هر prototype: (Q, n_way)
    dists = torch.cdist(query_emb, prototypes, p=2)
    logits = -dists  # فاصله کمتر = شباهت بیشتر = logit بزرگ‌تر

    loss = F.cross_entropy(logits, query_labels)
    preds = logits.argmax(dim=1)
    acc = (preds == query_labels).float().mean().item()

    return loss, acc
