from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from torchvision.transforms.functional import to_pil_image

from .evaluation import extract_embeddings


def plot_tsne_embeddings(
    embeddings,
    labels,
    label_map: Dict[int, str],
    name: str | None = None,
    num_class_show: int = 0,
):
    """
    Creates a two dimensional t SNE embedding plot of image features. 
    It is used to show how samples from different classes cluster in the learned feature space.
    """
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    if num_class_show > 0:
        selected_classes = sorted(np.unique(labels))[:num_class_show]
        mask = np.isin(labels, selected_classes)
        embeddings = embeddings[mask]
        labels = labels[mask]

    tsne = TSNE(n_components=2, init="random", random_state=42)
    reduced = tsne.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_classes)]

    plt.figure(figsize=(6, 6))

    for idx, label in enumerate(unique_labels):
        inds = np.where(labels == label)[0]
        plt.scatter(
            reduced[inds, 0],
            reduced[inds, 1],
            color=colors[idx],
            alpha=0.6,
            label=label_map.get(label, f"Class {label}"),
            s=12,
        )

        centroid = reduced[inds].mean(axis=0)
        class_name = label_map.get(label, f"Class {label}")
        plt.text(
            centroid[0],
            centroid[1],
            class_name,
            fontsize=6,
            ha="center",
            va="center",
            weight="bold",
        )

    title = (
        f"t-SNE - {name}"
        if num_class_show == 0
        else f"t-SNE (Top {num_class_show} classes {name})"
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    test_labels,
    predicted_labels,
    map: Dict[int, str] | None = None,
    name: str | None = None,
    num_class_show: int = 0,
):
    """
    Confusion-matrix layout preserved from Task 4.
    """
    test_labels = np.array(test_labels)
    predicted_labels = np.array(predicted_labels)

    if num_class_show > 0:
        selected_classes = sorted(np.unique(test_labels))[:num_class_show]
        mask = np.isin(test_labels, selected_classes)
        test_labels = test_labels[mask]
        predicted_labels = predicted_labels[mask]
    else:
        selected_classes = sorted(np.unique(test_labels))

    target_names = [map[i] if map else f"Class {i}" for i in selected_classes]

    report_dict = classification_report(
        test_labels,
        predicted_labels,
        labels=selected_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    _ = report_dict

    cm_ = confusion_matrix(test_labels, predicted_labels, labels=selected_classes)
    cm_percent = cm_.astype("float") / cm_.sum(axis=1, keepdims=True) * 100
    cm_percent_rounded = np.rint(cm_percent).astype(int)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm_percent_rounded,
        xticklabels=target_names,
        yticklabels=target_names,
        cmap="YlGnBu",
        annot=False,
        fmt="d",
    )

    for i in range(cm_.shape[0]):
        for j in range(cm_.shape[1]):
            val = cm_percent_rounded[i, j]
            if val > 0:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val}%",
                    ha="center",
                    va="center",
                    color="white" if val > 50 else "black",
                    fontsize=11,
                )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({name})")
    plt.tight_layout()
    plt.show()


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    # Denormalisation reverses the preprocessing so that matplotlib receives pixel intensities in the expected range
    return tensor * std + mean


def show_top3_recommendations_same_class(
    model,
    test_loader,
    support_loader,
    test_dataset,
    support_dataset,
    label_map: Dict[int, str] | None = None,
    n_samples: int = 5,
    device: str = "cuda",
    confidence_threshold: float = 0.57,
):
    """
    Shows for each selected query image up to three similar items from a support set of images. 
    The display groups each query with its closest neighbors in the learned embedding space and annotates the examples with a confidence estimate.
    """
    support_embeddings, support_labels = extract_embeddings(
        support_loader, model, device=device
    )
    test_embeddings, test_labels = extract_embeddings(
        test_loader, model, device=device
    )

    support_labels = torch.tensor(support_labels).cpu()
    test_labels = torch.tensor(test_labels)

    # Normalised embeddings allow cosine similarity to act as a pure inner product downstream
    support_embeddings = F.normalize(torch.tensor(support_embeddings), dim=1).to(device)
    test_embeddings = F.normalize(torch.tensor(test_embeddings), dim=1).to(device)

    num_test = test_embeddings.shape[0]
    # Random sampling keeps the visualisation diverse between notebook executions
    indices_ = torch.randperm(num_test)[:n_samples]
    test_embeddings_subset = test_embeddings[indices_]
    test_labels_subset = test_labels[indices_]
    test_samples_subset = [test_dataset[i.item()] for i in indices_]

    similarities = torch.matmul(test_embeddings_subset, support_embeddings.T)
    k = 3
    top_scores, top_indices = torch.topk(similarities, k=k, dim=1)
    top_indices = top_indices.cpu()
    top_labels = support_labels[top_indices]

    # Confidence value summarises how close the query is to its retrieved neighbours which guides selective display
    confidences = top_scores.mean(axis=1)

    shown = 0
    plt.figure(figsize=(10, 2.5 * n_samples))

    for i in range(len(test_labels_subset)):
        confidence = confidences[i]
        indices = top_indices[i]
        scores = top_scores[i]
        test_img, _ = test_samples_subset[i]
        test_img = to_pil_image(
            denormalize(
                test_img,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

        plt.subplot(n_samples, 4, 4 * shown + 1)
        plt.imshow(test_img)
        class_idx = int(test_labels_subset[i])
        class_name = (
            label_map.get(class_idx, f"Class {class_idx}")
            if label_map
            else f"Class {class_idx}"
        )

        if confidence >= confidence_threshold:
            plt.title(f"{class_name}\nConfidence: {confidence*100:.0f}%")
        else:
            plt.title(f"{class_name}\n⚠ Low Confidence: {confidence*100:.0f}%")
        plt.axis("off")

        if confidence >= confidence_threshold:
            for j in range(3):
                support_img, _ = support_dataset[indices[j].item()]
                support_img = to_pil_image(
                    denormalize(
                        support_img,
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                )
                plt.subplot(n_samples, 4, 4 * shown + 2 + j)
                plt.imshow(support_img)
                plt.title(f"Top-{j+1}")
                plt.axis("off")

        shown += 1

    plt.tight_layout()
    plt.show()


def visualize_shirt_tshirt_transition(
    support_loader,
    test_loader,
    support_dataset,
    test_dataset,
    shirt_label: int,
    tshirt_label: int,
    model,
    n_steps: int = 7,
    title: str = "Transition: Shirt ↔ T-shirt",
    device: str = "cuda",
):
    """
    Visualizes how the network interpolates between the shirt and t shirt classes in the embedding space. 
    It selects typical examples for each class and then shows images whose embeddings lie along the straight path between the two centroids for both train and test sets.
    """

    def compute_centroid(embs: torch.Tensor) -> torch.Tensor:
        return embs.mean(axis=0)

    def get_closest_index(embs: torch.Tensor, target_vec: torch.Tensor) -> int:
        embs_np = embs.cpu().numpy()
        target_np = target_vec.cpu().numpy()
        dists = np.linalg.norm(embs_np - target_np, axis=1)
        return int(np.argmin(dists))

    def filter_indices(embedding: torch.Tensor, labels: torch.Tensor):
        shirt_inds = np.where(labels == shirt_label)[0]
        tshirt_inds = np.where(labels == tshirt_label)[0]
        shirt_embs = embedding[shirt_inds]
        tshirt_embs = embedding[tshirt_inds]

        shirt_centroid = compute_centroid(shirt_embs)
        tshirt_centroid = compute_centroid(tshirt_embs)

        typical_shirt_idx = shirt_inds[
            get_closest_index(shirt_embs, shirt_centroid)
        ]
        typical_tshirt_idx = tshirt_inds[
            get_closest_index(tshirt_embs, tshirt_centroid)
        ]

        interpolated_points = [
            (1 - alpha) * shirt_centroid + alpha * tshirt_centroid
            for alpha in np.linspace(0, 1, n_steps)
        ]
        interpolated_points = [point.detach().cpu() for point in interpolated_points]
        transition_indices = [
            get_closest_index(embedding, point) for point in interpolated_points
        ]

        return typical_shirt_idx, typical_tshirt_idx, transition_indices

    def make_unified_plot(
        support_dataset,
        test_dataset,
        n_steps,
        support_typical_shirt_idx,
        support_typical_tshirt_idx,
        support_transition_indices,
        test_typical_shirt_idx,
        test_typical_tshirt_idx,
        test_transition_indices,
        title: str = "Shirt ↔ T-shirt Transition",
    ):
        cols = n_steps + 2
        plt.figure(figsize=(2.5 * cols, 6))

        def plot_image(ax_idx, idx, row, label, dataset):
            img, _ = dataset[idx]
            img = to_pil_image(
                denormalize(
                    img,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
            plt.subplot(2, cols, ax_idx + row * cols + 1)
            plt.imshow(img)
            plt.title(label, fontsize=10)
            plt.axis("off")

        for row, (
            typical_shirt_idx,
            typical_tshirt_idx,
            transition_indices,
            name,
            dataset,
        ) in enumerate(
            [
                (
                    support_typical_shirt_idx,
                    support_typical_tshirt_idx,
                    support_transition_indices,
                    "Train",
                    support_dataset,
                ),
                (
                    test_typical_shirt_idx,
                    test_typical_tshirt_idx,
                    test_transition_indices,
                    "Test",
                    test_dataset,
                ),
            ]
        ):
            plot_image(0, typical_shirt_idx, row, f"Typical\nShirt ({name})", dataset)
            for i, idx in enumerate(transition_indices):
                alpha = i / (n_steps - 1) if n_steps > 1 else 0
                plot_image(
                    i + 1,
                    idx,
                    row,
                    f"α={alpha:.2f}",
                    dataset,
                )
            plot_image(
                n_steps + 1,
                typical_tshirt_idx,
                row,
                f"Typical\nT-shirt ({name})",
                dataset,
            )

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    support_embeddings, support_labels = extract_embeddings(
        support_loader, model, device=device
    )
    test_embeddings, test_labels = extract_embeddings(
        test_loader, model, device=device
    )

    support_labels = torch.tensor(support_labels)
    test_labels = torch.tensor(test_labels)

    support_embeddings = F.normalize(torch.tensor(support_embeddings), dim=1).to(device)
    test_embeddings = F.normalize(torch.tensor(test_embeddings), dim=1).to(device)

    (
        support_typical_shirt_idx,
        support_typical_tshirt_idx,
        support_transition_indices,
    ) = filter_indices(support_embeddings, support_labels)
    (
        test_typical_shirt_idx,
        test_typical_tshirt_idx,
        test_transition_indices,
    ) = filter_indices(test_embeddings, test_labels)

    make_unified_plot(
        support_dataset,
        test_dataset,
        n_steps,
        support_typical_shirt_idx,
        support_typical_tshirt_idx,
        support_transition_indices,
        test_typical_shirt_idx,
        test_typical_tshirt_idx,
        test_transition_indices,
        title=title,
    )
