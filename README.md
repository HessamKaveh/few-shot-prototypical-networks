# Few-Shot Learning with Prototypical Networks

A from-scratch implementation of Prototypical Networks for few-shot image
classification, trained and evaluated on the Omniglot dataset.

## Task Setup
5-way 5-shot learning: for each episode, 5 random character classes are
sampled; the model sees only 5 labeled examples per class (support set)
and must classify 5 new unlabeled examples per class (query set).

## Pipeline
1. Episodic sampling: each training step is a new N-way K-shot task
2. Embed all support and query images with a shared CNN encoder
3. Compute class prototypes = mean embedding of each class's support examples
4. Classify queries by nearest prototype (Euclidean distance)
5. Evaluate generalization on entirely unseen character classes

## Key Idea
Unlike standard classifiers, the model never sees a fixed set of classes —
it learns a general embedding space where same-class images cluster together,
allowing classification of brand-new classes from just a few examples.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/evaluate.py
```

## Results
- `results/training_curves.png` — loss and accuracy (seen vs. unseen classes)
- `results/episode_visualization.png` — a sample 5-way 5-shot episode

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
