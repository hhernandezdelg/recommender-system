from typing import Tuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.stats import mode
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def extract_embeddings(
    dataloader,
    model,
    embedding_dim: int = 8192,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the model on an entire data loader and collects a matrix of feature embeddings. 
    It also returns the associated labels so that the embeddings can be used for downstream analysis and k nearest neighbor evaluation.
    """
    model.eval()
    embeddings = np.zeros((len(dataloader.dataset), embedding_dim))
    labels = np.zeros(len(dataloader.dataset))
    k = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            emb = model.get_embedding(images).cpu().numpy()
            embeddings[k : k + len(images)] = emb
            labels[k : k + len(images)] = targets.numpy()
            k += len(images)

    return embeddings, labels


def evaluate_knn(
    support_loader,
    test_loader,
    model,
    k: int = 5,
    device: str = "cuda",
    name: str | None = None,
):
    """
    kNN evaluation in embedding space, as in Task 2.
    Returns (test_labels, predicted_labels).
    """
    support_embeddings, support_labels = extract_embeddings(
        support_loader, model, device=device
    )
    test_embeddings, test_labels = extract_embeddings(
        test_loader, model, device=device
    )

    support_embeddings = F.normalize(torch.tensor(support_embeddings), dim=1)
    test_embeddings = F.normalize(torch.tensor(test_embeddings), dim=1)

    sim_matrix = torch.matmul(test_embeddings, support_embeddings.T)
    topk_indices = torch.topk(sim_matrix, k=k, dim=1).indices
    topk_labels = support_labels[topk_indices]
    predicted_labels, _ = mode(topk_labels, axis=1)
    predicted_labels = predicted_labels.squeeze()

    return test_labels, predicted_labels


def compute_correct_counts(
    sim_matrix: torch.Tensor,
    test_labels: Sequence[int],
    support_labels: Sequence[int],
    k: int = 3,
):
    """
    Computes, for each query, how many of the top-k matches have the correct class.
    """
    topk_scores, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
    device = sim_matrix.device
    support_labels = torch.tensor(support_labels).to(device)
    test_labels = torch.tensor(test_labels).to(device)
    topk_labels = support_labels[topk_indices]

    correct_counts = (topk_labels == test_labels.unsqueeze(1)).sum(dim=1)
    return correct_counts.cpu().numpy(), topk_scores.cpu().numpy()


def compute_coverage_error_from_counts(
    confidences: np.ndarray,
    correct_counts: np.ndarray,
    taus: np.ndarray | None = None,
):
    """
    Computes coverage and error-rate curves as a function of confidence threshold.
    """
    if taus is None:
        taus = np.linspace(0, 1, 50)

    coverages = []
    error_rates = []
    used_thresholds = []

    total_samples = len(confidences)

    for tau in taus:
        mask = confidences >= tau
        if mask.sum() == 0:
            continue

        shown_counts = correct_counts[mask]

        coverage = mask.sum() / total_samples
        error = np.sum(shown_counts == 0) / len(shown_counts)

        coverages.append(coverage * 100)
        error_rates.append(error * 100)
        used_thresholds.append(tau)

    return np.array(coverages), np.array(error_rates), used_thresholds


def plot_error_coverage_curve(
    model,
    test_loader,
    support_loader,
    device: str = "cuda",
    k: int = 3,
):
    """
    Error vs coverage plot, with identical layout to Task 3.
    """
    support_embs, support_labels = extract_embeddings(
        support_loader, model, device=device
    )
    test_embs, test_labels = extract_embeddings(test_loader, model, device=device)
    support_embs = F.normalize(torch.tensor(support_embs), dim=1).to(device)
    test_embs = F.normalize(torch.tensor(test_embs), dim=1).to(device)
    sim_matrix = torch.matmul(test_embs, support_embs.T)

    correct_counts, top_scores = compute_correct_counts(
        sim_matrix, test_labels, support_labels, k=k
    )
    confidences = top_scores.mean(axis=1)
    coverages, error_rates, taus = compute_coverage_error_from_counts(
        confidences, correct_counts
    )

    taus = np.array(taus)
    norm = colors.Normalize(vmin=0.5, vmax=1)
    plt.figure(figsize=(6, 5))
    plt.plot(
        error_rates,
        coverages,
        label="avg_topk confidence",
        color="blue",
        linewidth=0.5,
    )
    sc = plt.scatter(
        error_rates,
        coverages,
        c=taus,
        cmap="managua",
        norm=norm,
        s=80,
        label="τ points",
    )

    for err, cov, tau in zip(error_rates, coverages, taus):
        if np.round(tau, 2) in [0.9, 0.8, 0.7]:
            plt.annotate(
                f"τ={tau:.2f}",
                (err, cov),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )

    mask_2 = np.isclose(np.array(error_rates), 10.0, atol=0.2)

    if mask_2.any():
        best_idx_2 = np.argmax(np.array(coverages)[mask_2])
        error_2 = np.array(error_rates)[mask_2][best_idx_2]
        cov_2 = np.array(coverages)[mask_2][best_idx_2]
        tau_2 = taus[mask_2][best_idx_2]

        plt.scatter(
            [error_2],
            [cov_2],
            s=60,
            color="green",
            edgecolor="black",
            label=f"@10% error: cov≈{cov_2:.1f}%",
        )
        plt.axhline(cov_2, color="green", linestyle="--", linewidth=1)
        plt.axvline(error_2, color="green", linestyle="--", linewidth=1)

    mask = np.array(error_rates) <= 10
    if mask.any():
        best_idx = np.argmax(np.array(coverages)[mask])
        best_error = np.array(error_rates)[mask][best_idx]
        best_cov = np.array(coverages)[mask][best_idx]
        best_tau = taus[mask][best_idx]

        plt.scatter(
            [best_error],
            [best_cov],
            s=60,
            color="red",
            edgecolor="black",
            label=f"Best τ={best_tau:.2f},\ncov≈{best_cov:.1f}%, err≈{best_error:.1f}%",
        )

    plt.xlabel("Error Rate (%)")
    plt.ylabel("Coverage (%)")
    plt.title("Error vs Coverage Curve")
    plt.colorbar(sc, label="Confidence Threshold (τ)")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
