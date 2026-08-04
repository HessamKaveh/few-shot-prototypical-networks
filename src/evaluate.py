import os
import torch
import matplotlib.pyplot as plt

from dataset import EpisodicOmniglot
from model import EmbeddingNet
from prototypical import compute_prototypes

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_embedding_net.pt")

N_WAY = 5
K_SHOT = 5
Q_QUERY = 5


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_data = EpisodicOmniglot(background=False, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY)

    model = EmbeddingNet(in_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    s_img, s_label, q_img, q_label = val_data.sample_episode()
    s_img, s_label = s_img.to(device), s_label.to(device)
    q_img, q_label = q_img.to(device), q_label.to(device)

    with torch.no_grad():
        s_emb = model(s_img)
        q_emb = model(q_img)
        prototypes = compute_prototypes(s_emb, s_label, N_WAY)
        dists = torch.cdist(q_emb, prototypes, p=2)
        preds = (-dists).argmax(dim=1)

    # نمایش support set (نمونه‌های آموزشی هر کلاس)
    fig, axes = plt.subplots(N_WAY, K_SHOT + 1, figsize=(2 * (K_SHOT + 1), 2 * N_WAY))
    for c in range(N_WAY):
        class_imgs = s_img[s_label == c].cpu()
        for k in range(K_SHOT):
            axes[c, k].imshow(class_imgs[k, 0], cmap="gray")
            axes[c, k].axis("off")
            if k == 0:
                axes[c, k].set_ylabel(f"Class {c}", fontsize=10)

        # یک نمونه query صحیح/غلط برای این کلاس نشون بده
        query_idx = (q_label == c).nonzero()[0].item()
        q_pred = preds[query_idx].item()
        correct = "✓" if q_pred == c else "✗"
        axes[c, K_SHOT].imshow(q_img[query_idx, 0].cpu(), cmap="gray")
        axes[c, K_SHOT].set_title(f"Query pred={q_pred} {correct}", fontsize=9)
        axes[c, K_SHOT].axis("off")

    plt.suptitle(f"{N_WAY}-way {K_SHOT}-shot Episode (unseen classes)")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "episode_visualization.png")
    plt.savefig(out_path, dpi=150)

    overall_acc = (preds == q_label).float().mean().item()
    print(f"Sample episode accuracy: {overall_acc:.4f}")
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    evaluate()
