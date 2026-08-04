import os
import random
import torch
from collections import defaultdict
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

IMG_SIZE = 28

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class EpisodicOmniglot:
    """
    برای Few-Shot Learning، داده‌ها به شکل 'episode' نمونه‌گیری می‌شن:
    هر episode شامل n_way کلاس تصادفی است، که هرکدام:
    - k_shot نمونه برای support set (یادگیری)
    - q_query نمونه برای query set (ارزیابی) دارند
    """
    def __init__(self, background=True, n_way=5, k_shot=5, q_query=5):
        self.dataset = datasets.Omniglot(
            root=DATA_DIR, background=background, download=True, transform=transform
        )
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # گروه‌بندی ایندکس‌ها بر اساس کلاس (character)
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset._flat_character_images):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

    def sample_episode(self):
        episode_classes = random.sample(self.classes, self.n_way)

        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for new_label, cls in enumerate(episode_classes):
            indices = random.sample(self.class_to_indices[cls], self.k_shot + self.q_query)
            support_idx = indices[:self.k_shot]
            query_idx = indices[self.k_shot:]

            for idx in support_idx:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(new_label)

            for idx in query_idx:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(new_label)

        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels
