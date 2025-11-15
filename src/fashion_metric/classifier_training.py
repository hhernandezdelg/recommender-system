import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from IPython.display import clear_output
from matplotlib import pyplot as plt


class TransferLearnP2(nn.Module):
    """
    Training wrapper that handles optimization, metric tracking, and plotting
    for supervised classification models.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        model_name: str = "default_model",
        lr: float = 1e-4,
    ):
        super().__init__()
        self.model = base_model
        self.lr = lr
        self.num_classes = num_classes
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()

        # Metric tracking
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.train_aucs, self.val_aucs = [], []
        self.train_balanced_accs, self.val_balanced_accs = [], []

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr: float | None = None):
        lr = lr if lr is not None else self.lr
        if lr is None:
            raise ValueError("Learning rate is not defined")
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

    def train_one_epoch(self, dataloader: DataLoader, optimizer, device):
        self.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        probs_all = []
        targets_all = []

        for x, y in dataloader:
            # Move batch to the target device and reset gradients
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass and loss computation
            logits = self(x)
            loss = self.criterion(logits, y)
            loss.backward()
            optimizer.step()

            # Batch-level statistics that feed the epoch aggregates
            epoch_loss += loss.item() * x.size(0)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            probs_all.append(probs.detach().cpu())
            targets_all.append(y.detach().cpu())

        all_probs = torch.cat(probs_all)
        all_targets = torch.cat(targets_all)
        all_preds = all_probs.argmax(dim=1)

        avg_loss = epoch_loss / total
        acc = correct / total

        balanced_acc = balanced_accuracy_score(
            all_targets.numpy(), all_preds.numpy()
        )

        try:
            auc = roc_auc_score(
                all_targets, all_probs, multi_class="ovr", average="macro"
            )
        except Exception:
            auc = float("nan")

        self.train_losses.append(avg_loss)
        self.train_accs.append(acc)
        self.train_balanced_accs.append(balanced_acc)
        self.train_aucs.append(auc)

        return avg_loss, acc, balanced_acc, auc

    def validate_one_epoch(self, dataloader: DataLoader, device):
        self.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0
        probs_all = []
        targets_all = []

        with torch.no_grad():
            # Validation mirrors training but without gradient computation
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits = self(x)
                loss = self.criterion(logits, y)

                epoch_loss += loss.item() * x.size(0)
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                probs_all.append(probs.cpu())
                targets_all.append(y.cpu())

        all_probs = torch.cat(probs_all)
        all_targets = torch.cat(targets_all)
        all_preds = all_probs.argmax(dim=1)

        avg_loss = epoch_loss / total
        acc = correct / total

        # Compute balanced accuracy
        balanced_acc = balanced_accuracy_score(
            all_targets.numpy(), all_preds.numpy()
        )

        try:
            auc = roc_auc_score(
                all_targets, all_probs, multi_class="ovr", average="macro"
            )
        except Exception as e:
            print(f"Could not compute val AUC: {e}")
            auc = float("nan")

        self.val_losses.append(avg_loss)
        self.val_accs.append(acc)
        self.val_balanced_accs.append(balanced_acc)
        self.val_aucs.append(auc)

        return avg_loss, acc, balanced_acc, auc

    def plot_metrics(
        self,
        save_dir: str | Path = "./output",
        filename: str = "final_metrics.png",
        test_loss: float | None = None,
        test_acc: float | None = None,
        test_bal_acc: float | None = None,
    ):
        # live plotting of training/validation metrics
        clear_output(wait=True)
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(16, 5))

        train_color = "#1f77b4"
        val_color = "#ff7f0e"
        test_color = "#2ca02c"

        # Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss", color=train_color)
        if len(self.val_losses) > 0:
            plt.plot(epochs, self.val_losses, label="Val Loss", color=val_color)
        if test_loss is not None:
            plt.scatter(
                epochs[-1],
                test_loss,
                color=test_color,
                s=80,
                marker="*",
                label="Test Loss",
                edgecolors="black",
                zorder=5,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accs, label="Train Acc", color=train_color)
        if len(self.val_accs) > 0:
            plt.plot(epochs, self.val_accs, label="Val Acc", color=val_color)
        if test_acc is not None:
            plt.scatter(
                epochs[-1],
                test_acc,
                color=test_color,
                s=80,
                marker="*",
                label="Test Acc",
                edgecolors="black",
                zorder=5,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Balanced Accuracy
        plt.subplot(1, 3, 3)
        plt.plot(
            epochs,
            self.train_balanced_accs,
            label="Train Bal Acc",
            color=train_color,
        )
        if len(self.val_balanced_accs) > 0:
            plt.plot(
                epochs,
                self.val_balanced_accs,
                label="Val Bal Acc",
                color=val_color,
            )
        if test_bal_acc is not None:
            plt.scatter(
                epochs[-1],
                test_bal_acc,
                color=test_color,
                s=80,
                marker="*",
                label="Test Bal Acc",
                edgecolors="black",
                zorder=5,
            )
        plt.xlabel("Epoch")
        plt.ylabel("Balanced Accuracy")
        plt.legend()

        # Metrics for each model are saved in a dedicated subdirectory so that cross validation folds do not overwrite each other
        save_dir_ = Path(save_dir) / self.model_name
        save_dir_.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_dir_ / filename, bbox_inches="tight")
        plt.show()


def train_transfer_model(
    config: Dict[str, Any],
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    save_dir: str | Path = "./output",
    fold: int | None = None,
) -> Dict[str, Any]:
    """
    Trains a classification model and evaluates it on a held out test set. 
    The function logs metrics, plots learning curves and saves the trained weights and statistics to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    num_classes = config.get("num_classes", 39)

    # Initialize model and trainer
    base_model = config["model_fn"](num_classes)

    # Create trainer
    trainer = TransferLearnP2(
        base_model,
        num_classes=num_classes,
        model_name=model_name,
        lr=config["lr"],
    )
    trainer.to(device)
    optimizer = trainer.get_optimizer()

    for epoch in range(epochs):
        # Standard training loop
        train_loss, train_acc, train_bal_acc, _ = trainer.train_one_epoch(
            train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc, val_bal_acc, _ = trainer.validate_one_epoch(
            val_loader,
            device=device,
        )

        trainer.plot_metrics(
            save_dir=save_dir,
            filename="live_metrics.png",
        )

    # Evaluate on the held-out test set
    trainer.to(device)
    trainer.eval()
    test_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        # Collect predictions, probabilities, and targets to compute metrics
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = trainer(X_batch)
            loss = trainer.criterion(outputs, y_batch)

            test_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

            probs = F.softmax(outputs, dim=1)

            all_preds.append(predicted.cpu())
            all_targets.append(y_batch.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)

    test_loss /= total
    test_acc = correct / total
    test_balanced_acc = balanced_accuracy_score(
        all_targets.numpy(), all_preds.numpy()
    )

    print(
        f"Test → Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Balanced Acc: {test_balanced_acc:.4f}"
    )

    trainer.plot_metrics(
        save_dir=save_dir,
        filename="final_metrics.png",
        test_loss=test_loss,
        test_acc=test_acc,
        test_bal_acc=test_balanced_acc,
    )

    # The metrics dictionary contains full training curves as well as the scalar values used for reporting
    metrics = {
        "train_loss": trainer.train_losses,
        "train_acc": trainer.train_accs,
        "train_bal_acc": trainer.train_balanced_accs,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_bal_acc": test_balanced_acc,
        "val_loss": trainer.val_losses,
        "val_acc": trainer.val_accs,
        "val_bal_acc": trainer.val_balanced_accs,
    }

    if fold is not None:
        weight_filename = f"{model_name}_fold{fold}.pth"
    else:
        weight_filename = f"{model_name}.pth"

    torch.save(
        trainer.model.state_dict(),
        os.path.join(save_dir, weight_filename),
    )

    metrics_filename = (
        f"metrics_fold{fold}.json" if fold is not None else "metrics.json"
    )
    with open(os.path.join(save_dir, metrics_filename), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {os.path.join(save_dir, metrics_filename)}")

    print(
        f"Train → Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Balanced Acc: {train_bal_acc:.4f}"
    )
    print(
        f"Val → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Balanced Acc: {val_bal_acc:.4f}"
    )
    print(
        f"Test → Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Balanced Acc: {test_balanced_acc:.4f}"
    )

    return metrics
