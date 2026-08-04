import os
import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import EpisodicOmniglot
from model import EmbeddingNet
from prototypical import prototypical_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_WAY = 5
K_SHOT = 5
Q_QUERY = 5
NUM_EPISODES_TRAIN = 2000
NUM_EPISODES_VAL = 300


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Setting: {N_WAY}-way {K_SHOT}-shot learning")

    train_data = EpisodicOmniglot(background=True, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY)
    val_data = EpisodicOmniglot(background=False, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY)

    model = EmbeddingNet(in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    eval_every = 100
    best_val_acc = 0.0

    running_loss, running_acc = 0.0, 0.0

    for episode in range(1, NUM_EPISODES_TRAIN + 1):
        model.train()
        s_img, s_label, q_img, q_label = train_data.sample_episode()
        s_img, s_label = s_img.to(device), s_label.to(device)
        q_img, q_label = q_img.to(device), q_label.to(device)

        optimizer.zero_grad()
        s_emb = model(s_img)
        q_emb = model(q_img)
        loss, acc = prototypical_loss(s_emb, s_label, q_emb, q_label, N_WAY)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc

        if episode % eval_every == 0:
            avg_loss = running_loss / eval_every
            avg_acc = running_acc / eval_every
            running_loss, running_acc = 0.0, 0.0

            # ─── ارزیابی روی کلاس‌های دیده‌نشده (val) ───
            model.eval()
            val_accs = []
            with torch.no_grad():
                for _ in range(NUM_EPISODES_VAL // eval_every):
                    s_img, s_label, q_img, q_label = val_data.sample_episode()
                    s_img, s_label = s_img.to(device), s_label.to(device)
                    q_img, q_label = q_img.to(device), q_label.to(device)
                    s_emb = model(s_img)
                    q_emb = model(q_img)
                    _, val_acc = prototypical_loss(s_emb, s_label, q_emb, q_label, N_WAY)
                    val_accs.append(val_acc)

            avg_val_acc = sum(val_accs) / len(val_accs)

            history["train_loss"].append(avg_loss)
            history["train_acc"].append(avg_acc)
            history["val_acc"].append(avg_val_acc)

            print(f"Episode {episode}/{NUM_EPISODES_TRAIN} | "
                  f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | Val Acc: {avg_val_acc:.4f}")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_embedding_net.pt"))

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"])
    plt.title("Prototypical Loss"); plt.xlabel(f"x{eval_every} episodes")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc (seen classes)")
    plt.plot(history["val_acc"], label="Val Acc (unseen classes)")
    plt.legend(); plt.title(f"{N_WAY}-way {K_SHOT}-shot Accuracy")
    plt.xlabel(f"x{eval_every} episodes")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
    print(f"\nBest validation accuracy (unseen classes): {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
