from itertools import combinations
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import BatchSampler
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from IPython.display import clear_output


class BalancedBatchSampler(BatchSampler):
    """
    Returns batches of size n_classes * n_samples.
    Each batch contains n_classes different classes and n_samples samples for each class.
    Used for training with triplet loss.
    """

    def __init__(self, labels: Iterable[int], n_classes: int, n_samples: int):
        self.labels = list(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {
            label: np.where(np.array(self.labels) == label)[0]
            for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[class_] : self.used_label_indices_count[
                            class_
                        ]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if (
                    self.used_label_indices_count[class_] + self.n_samples
                    > len(self.label_to_indices[class_])
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self) -> int:
        return self.n_dataset // self.batch_size


class TripletLoss(nn.Module):
    """
    Triplet loss with semi hard negative mining.
    Encourages the model to learn embeddings where an anchor is closer to positive examples than to negative examples by at least a margin.
    """

    def __init__(self, margin: float, triplet_selector: "InformativeNegativeTripletSelector"):
        super().__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings: torch.Tensor, target: torch.Tensor):
        # Select triplets using the provided triplet selector
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            # Move triplets to GPU if embeddings are on GPU
            triplets = triplets.cuda()

        # Unpack triplet indices
        anchor_idx = triplets[:, 0]
        positive_idx = triplets[:, 1]
        negative_idx = triplets[:, 2]

        # Compute squared distances for positive and negative pairs
        dist_pos = (embeddings[anchor_idx] - embeddings[positive_idx]).pow(2).sum(1)
        dist_neg = (embeddings[anchor_idx] - embeddings[negative_idx]).pow(2).sum(1)

        # Compute triplet loss with margin
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()


def pdist(vectors: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise squared Euclidean distance matrix for a set of vectors.
    """
    distance_matrix = (
        -2 * vectors.mm(torch.t(vectors))
        + vectors.pow(2).sum(dim=1).view(1, -1)
        + vectors.pow(2).sum(dim=1).view(-1, 1)
    )
    return distance_matrix


class InformativeNegativeTripletSelector:
    """
    Implements a strategy to mine informative negative examples for triplet loss training. 
    It prefers semi hard negatives when they exist and falls back to the hardest negatives otherwise.
    Semi hard negatives are those that are farther from the anchor than the positive but still within the margin.
    Refer to the docs/ for more details on semi hard negative and hard negative mining.
    """

    def __init__(self, margin: float):
        self.margin = margin

    def get_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor):
        if not torch.cuda.is_available():
            embeddings = embeddings.cpu()
        
        # Compute pairwise distance matrix
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        # Convert labels to numpy for easier indexing
        labels_np = labels.cpu().data.numpy()
        
        triplets = []
        # Iterate over each unique label to form anchor-positive pairs
        for label in set(labels_np):
            # Find indices for the current label
            label_mask = labels_np == label

            # Get all indices for this label
            label_indices = np.where(label_mask)[0]
            
            # Skip if less than two examples for this label
            if len(label_indices) < 2:
                continue

            # Get all indices for negative examples
            negative_indices = np.where(np.logical_not(label_mask))[0]

            # Generate all possible anchor-positive pairs for this label
            anchor_positives = list(combinations(label_indices, 2))

            # Convert to numpy array for easier indexing
            anchor_positives = np.array(anchor_positives)


            # Extract distances for anchor-positive and anchor-negative pairs
            anchor_indices = anchor_positives[:, 0]
            positive_indices = anchor_positives[:, 1]

            # Compute distances
            dist_pos = distance_matrix[anchor_indices, positive_indices]
            dist_neg = distance_matrix[anchor_indices][:, negative_indices]

            # Mask selects negatives that lie in the semi hard band relative to each anchor positive pair
            mask = (dist_neg > dist_pos[:, None]) & (
                dist_neg < (dist_pos + self.margin)[:, None]
            )

            # Get valid negative indices for each anchor-positive pair
            valid_idx = mask.nonzero(as_tuple=False)

            # For each anchor-positive pair, select a negative example
            for i in range(len(anchor_positives)):
                negs_i = valid_idx[valid_idx[:, 0] == i][:, 1]
                if len(negs_i) > 0:
                    # Semi hard negatives provide informative gradients without destabilising training
                    neg_idx = negative_indices[
                        negs_i[torch.randint(len(negs_i), ())]
                    ]
                else:
                    # When no semi hard negative exists we fall back to the hardest example to avoid wasting the triplet
                    neg_idx = negative_indices[dist_neg[i].argmin()]

                triplets.append(
                    [anchor_positives[i][0], anchor_positives[i][1], neg_idx]
                )

        return torch.LongTensor(np.array(triplets))


class Trainer:
    """
    Trainer class to handle the training loop for metric learning with triplet loss. 
    It manages training and validation phases, loss computation, backpropagation with mixed precision,
    and tracks metrics like Recall@k. It also provides live plotting of loss and recall during
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_DataLoader,
        validation_DataLoader,
        epochs: int,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.epochs = epochs
        self.best_val_loss = float("inf")
        self.scaler = GradScaler()

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.recall_history: list[float] = []

    def run_trainer(self):
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            train_losses = []
            for batch in self.training_DataLoader:
                # Balanced batches already contain several classes
                x, y = batch
                input, target = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass under autocast and compute triplet loss
                with autocast():
                    out = self.model(input, return_embeddings=True)
                    loss = self.criterion(out, target)

                # Backpropagation with gradient scaling for numerical stability
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_value = loss.item()
                train_losses.append(loss_value)

                # Explicit deletion helps prevent GPU memory fragmentation during long training runs
                del input, target, out, loss
                torch.cuda.empty_cache()

            # Validation phase records loss and Recall@k
            self.model.eval()
            valid_losses = []
            all_embeddings = []
            all_labels = []
            with torch.no_grad():
                for batch in self.validation_DataLoader:
                    x, y = batch
                    input, target = x.to(self.device), y.to(self.device)

                    out = self.model(input, return_embeddings=True)
                    loss = self.criterion(out, target)

                    loss_value = loss.item()
                    valid_losses.append(loss_value)

                    all_embeddings.append(out)
                    all_labels.append(target)

                    del input, target, out, loss_value
                    torch.cuda.empty_cache()

                embeddings = torch.cat(all_embeddings)
                labels = torch.cat(all_labels)

                recall = compute_recall_at_k(embeddings, labels, k=5)
                print(f"Recall@5: {recall:.4f}")

                print(
                    f"EPOCH: {epoch+1:0>{len(str(self.epochs))}}/{self.epochs}",
                    end=" ",
                )
                print(f"LOSS: {np.mean(train_losses):.4f}", end=" ")
                print(f"VAL-LOSS: {np.mean(valid_losses):.4f}", end="\n")

                torch.cuda.empty_cache()    # Clear GPU cache to free memory
                torch.cuda.ipc_collect()    # Collect inter-process GPU memory

            # Persist history for later plotting
            self.train_loss_history.append(float(np.mean(train_losses)))
            self.val_loss_history.append(float(np.mean(valid_losses)))
            self.recall_history.append(recall)
            plot_recall_loss_live(
                self.train_loss_history,
                self.val_loss_history,
                self.recall_history,
            )

        torch.save(self.model.state_dict(), "last_model.pt")
        print("📦 Saved final model as last_model.pt")


def plot_embeddings_live(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    recall: float | None = None,
    method: str = "pca",
    max_points: int = 1000,
):
    """
    Produces a two dimensional projection of embeddings using PCA or t SNE during training. 
    This helps monitor how the representation evolves and how classes separate over epochs.
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Subsample for visualization if too many points
    if len(embeddings_np) > max_points:
        idx = np.random.choice(len(embeddings_np), max_points, replace=False)
        embeddings_np = embeddings_np[idx]
        labels_np = labels_np[idx]

    if method == "tsne":
        reducer = TSNE(n_components=2, init="random", perplexity=30)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    reduced = reducer.fit_transform(embeddings_np)
    clear_output(wait=True)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels_np,
        cmap="tab10",
        s=10,
        alpha=0.8,
    )
    if recall is None:
        title = f"Epoch {epoch} - {method.upper()} Embedding"
    else:
        title = f"Epoch {epoch} - {method.upper()} Embedding\nRecall@5 = {recall:.4f}"
    plt.suptitle(title, fontsize=12)
    plt.colorbar(scatter, ticks=np.unique(labels_np))
    plt.tight_layout()
    plt.show()


def plot_recall_loss_live(
    train_loss_hist: list[float],
    val_loss_hist: list[float],
    recall_hist: list[float],
):
    """
    Plots the training and validation loss together with Recall at five over epochs. 
    It is used during training to track optimization progress and retrieval quality.
    """
    clear_output(wait=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall_hist, label="Recall@5")
    plt.xlabel("Epoch")
    plt.ylabel("Recall@5")
    plt.title("Recall@5 over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_recall_at_k(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
) -> float:
    """
    Computes Recall at k for normalized embeddings using cosine similarity. 
    The metric measures how often at least one of the k nearest neighbors has the same label as the query.
    """
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T

    # Exclude self-similarity by setting diagonal to -inf
    sim_matrix.fill_diagonal_(-float("inf"))

    # Get top k indices for each embedding
    topk_indices = torch.topk(sim_matrix, k=k, dim=1).indices

    # Check if any of the top k neighbors have the same label
    topk_labels = labels[topk_indices]

    # Determine if at least one of the top k labels matches the true label
    correct = (topk_labels == labels.unsqueeze(1)).any(dim=1)

    recall_at_k = correct.float().mean().item()
    return recall_at_k
